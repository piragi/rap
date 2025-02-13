import math
import random
from typing import Optional, Tuple

import numpy as np

from config import ModelParams
from token_tracker import TokenUsageStats
from tokenizer import Tokenizer
from weights import TransformerWeights
from rap.world_model import Action, State, predict_action, predict_state

W_EXP = 1.

class MCTSNode:
    """A node in the Monte Carlo Tree Search.
    
    Represents a state in the search tree, tracking visits, rewards, and maintaining 
    parent-child relationships. Each node corresponds to a step in the reasoning process.
    
    Attributes:
        state: Current state of reasoning
        action: Action that led to this state
        parent: Parent node in the tree
        visits: Number of times this node has been visited
        reward: Actual reward received at this node
        fast_reward: Initial reward estimate before full evaluation
        children: List of child nodes
        cum_rewards: Cumulative rewards for different paths
        max_reward: Maximum reward seen through this node
    """
    def __init__(self, state: Optional[State], action: Action, parent: Optional['MCTSNode'], reward: float, fast_reward: float):
        self.state = state
        self.action = action
        self.parent = parent
        self.visits = 0
        self.reward = reward
        self.fast_reward = fast_reward
        self.children: list[MCTSNode] = []
        self.cum_rewards: list[list[float]] = []
        self.max_reward = float('-inf')

    def is_terminal(self) -> bool:
        if self.state is not None and self.state.states:
            last_question = self.state.states[-1].subquestion.strip()
            if "Now we can answer the question:" in last_question:
                return True
        return False

    @property
    def q_value(self) -> float:
        """Calculate Q-value as max of average rewards in future steps"""
        if not self.cum_rewards:
            return self.fast_reward
        averages = [sum(reward_seq) / len(reward_seq) for reward_seq in self.cum_rewards]
        return max(averages) if averages else self.fast_reward

def calculate_reward(self_eval: float, confidence: float, alpha: float = 0.5) -> float:
    """Calculate reward using weighted geometric mean"""
    return math.pow(self_eval, alpha) * math.pow(confidence, 1 - alpha)

def select_node(node: MCTSNode) -> list[MCTSNode]:
    """Select a promising node to expand using UCT selection.
    
    Traverses the tree from root to leaf using Upper Confidence Bound (UCT)
    formula to balance exploration and exploitation.
    
    Args:
        node: Starting node for selection
        
    Returns:
        Path of nodes from root to selected leaf
    """
    path = []
    while True:
        path.append(node)
        if node.children is None or len(node.children) == 0 or node.is_terminal():
            return path

        node = max(
            node.children,
            key=lambda child:
            # For unvisited nodes, use fast_reward directly
            (child.fast_reward + W_EXP * np.sqrt(np.log(node.visits))) if child.visits == 0
            # For visited nodes, use standard UCT with max_reward
            else (child.max_reward + W_EXP * np.sqrt(np.log(node.visits) / child.visits)))

def simulation(node: MCTSNode,
               depth_limit: int,
               action_generation: int,
               tokenizer: Tokenizer,
               transformer_weights: TransformerWeights,
               model_params: ModelParams,
               confidence: int,
               token_stats: Optional[TokenUsageStats] = None) -> list[MCTSNode]:
    """Simulate a rollout from the given node to a terminal state or depth limit.
    
    Expands the tree and simulates a possible solution path, generating new actions
    and evaluating states along the way.
    
    Args:
        node: Starting node for simulation
        depth_limit: Maximum simulation depth
        action_generation: Number of actions to generate per step
        tokenizer: Tokenizer for model interactions
        transformer_weights: Model weights
        model_params: Model parameters
        confidence: Confidence threshold
        token_stats: Optional token usage tracker
        
    Returns:
        Path of nodes visited during simulation
    """
    path = []
    current_node = node

    while len(path) < depth_limit:
        if current_node not in path:
            path.append(current_node)
        if current_node.state is None:
            current_node.state = predict_state(current_node.parent.state,
                                               current_node.action,
                                               tokenizer,
                                               transformer_weights,
                                               model_params,
                                               confidence,
                                               token_stats=token_stats)
            current_node.reward = calculate_reward(current_node.fast_reward, current_node.state.states[-1].confidence)
        if current_node.is_terminal():
            break

        if not current_node.children:
            if len(path) == depth_limit - 1:
                original_question = "Now we can answer the question: " + current_node.state.question
                fast_reward = 1.0  # Or some other appropriate default
                current_node.children.append(MCTSNode(state=None, action=original_question, parent=current_node, reward=0., fast_reward=fast_reward))
            else:
                # Generate all actions at once
                actions_with_rewards = predict_action(current_node.state,
                                                      tokenizer,
                                                      transformer_weights,
                                                      model_params,
                                                      action_generation,
                                                      token_stats=token_stats)
                # Create children nodes
                for action, fast_reward in actions_with_rewards:
                    current_node.children.append(MCTSNode(state=None, action=action, parent=current_node, reward=0., fast_reward=fast_reward))

        # greedy selection vs random selection of children in simulation
        # current_node = max(current_node.children, key=lambda child: child.fast_reward)
        current_node = random.choice(current_node.children)
    return path

def backpropagation(path: list[MCTSNode]) -> float:
    """Update node statistics along the simulation path.
    
    Propagates rewards back up the tree and updates visit counts and maximum
    rewards for each node in the path.
    
    Args:
        path: List of nodes visited during simulation
        
    Returns:
        Total reward accumulated along the path
    """
    reward = 0
    for node in reversed(path):
        node.visits += 1
        reward = reward + node.reward
        position_from_end = len(path) - path.index(node)
        mean_reward = reward / position_from_end
        node.max_reward = max(node.max_reward, mean_reward)
    return reward

def get_highest_reward_path(root: MCTSNode) -> tuple[float, list[MCTSNode]]:
    def dfs(path: list[MCTSNode]) -> tuple[float, list[MCTSNode]]:
        current = path[-1]
        if current.is_terminal():
            if current.reward == 0:  # Check if reward has been evaluated
                return float('-inf'), path
            return current.max_reward, path

        if current.children is None:
            return float('-inf'), path

        # Look for children that have been evaluated
        evaluated_children = [c for c in current.children if c.reward != 0]
        if not evaluated_children:
            return float('-inf'), path

        return max((dfs(path + [child]) for child in evaluated_children), key=lambda x: x[0])

    return dfs([root])

def mcts(init_state: State,
         rollouts: int,
         depth_limit: int,
         action_generation: int,
         tokenizer: Tokenizer,
         transformer_weights: TransformerWeights,
         model_params: ModelParams,
         confidence: int,
         token_stats: Optional[TokenUsageStats] = None) -> tuple[State, MCTSNode]:
    """Execute Monte Carlo Tree Search to solve a math problem.
    
    Performs iterative search through possible reasoning steps to find the best path
    to solving the given math problem.
    
    Args:
        init_state: Initial problem state
        rollouts: Number of MCTS iterations to perform
        depth_limit: Maximum depth of the search tree
        action_generation: Number of candidate actions to generate at each step
        tokenizer: Tokenizer for model interactions
        transformer_weights: Model weights
        model_params: Model parameters
        confidence: Confidence threshold for predictions
        token_stats: Optional token usage tracker
        
    Returns:
        Tuple of (final state, root node) containing the best solution path
    """
    root = MCTSNode(init_state, init_state.question, None, 0., 0.)
    for _ in range(rollouts):
        path = select_node(root)
        last_node = path.pop()
        simulation_path = simulation(last_node,
                                     depth_limit - len(path),
                                     action_generation,
                                     tokenizer,
                                     transformer_weights,
                                     model_params,
                                     confidence,
                                     token_stats=token_stats)
        path.extend(simulation_path)
        backpropagation(path)

    reward, best_path = get_highest_reward_path(root)
    print_mcts(root)
    if reward == float('-inf'):
        print("No valid complete path found")
        return None
    append_best_path(best_path)
    return best_path[-1].state, root

def print_mcts(root: MCTSNode, prefix: str = "", is_last: bool = True, filename: str = "mcts_trace.txt"):
    """
    Print MCTS tree to both console and file, including detailed state information.
    Creates file if it doesn't exist.
    """
    # Format the node information
    connector = "└── " if is_last else "├── "
    if root.state is not None and len(root.state.states) != 0:
        # Get the last state's question and answer
        last_state = root.state.states[-1]
        node_info = (f"{prefix}{connector}[Visits: {root.visits}, "
                     f"Fast reward: {root.fast_reward:.2f}, "
                     f"Reward: {root.reward:.2f}\n"
                     f"{prefix}    Question: {last_state.subquestion}\n"
                     f"{prefix}    Answer: {last_state.subanswer}]")
    else:
        node_info = f"{prefix}{connector}[Visits: {root.visits}, Fast reward: {root.fast_reward:.2f}, Reward: {root.reward:.2f}]"

    # Print to console
    print(node_info)

    # Write to file
    with open(filename, 'a') as f:
        print(node_info, file=f)

    # Handle children
    child_prefix = prefix + ("    " if is_last else "│   ")
    if root.children:
        for i, child in enumerate(root.children):
            is_last_child = i == len(root.children) - 1
            print_mcts(child, child_prefix, is_last_child, filename)

def append_best_path(best_path: list[MCTSNode], filename: str = "mcts_trace.txt"):
    """Append the best path's questions and answers to the MCTS trace file."""
    with open(filename, 'a') as f:
        print("\nBEST PATH:", file=f)
        print("-" * 20, file=f)
        for node in best_path[1:]:  # Skip root node
            if node.state and node.state.states:
                last_state = node.state.states[-1]
                print(f"Question: {last_state.subquestion}, Reward: {node.reward:.2f}", file=f)
                print(f"Answer: {last_state.subanswer}", file=f)
                print("-" * 20, file=f)

def extract_answer(answer_text: str) -> Optional[float]:
    """Extract numerical answer from text."""
    try:
        if "The answer is" in answer_text:
            answer_str = answer_text.split("The answer is")[-1].strip().strip('.')
            answer_str = ''.join(c for c in answer_str if c.isdigit() or c == '.' or c == '-')
            return float(answer_str)
    except:
        pass
    return None
