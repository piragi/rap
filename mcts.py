import math
import random
from collections import defaultdict
from typing import Optional, Tuple

import numpy as np

from config import ModelParams
from token_tracker import TokenUsageStats
from tokenizer import Tokenizer
from weights import TransformerWeights
from world_model import Action, State, predict_action, predict_state

W_EXP = 1.

class MCTSNode:
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

def aggregate(root: MCTSNode, answer: float) -> Tuple[str, bool, float, float]:
    """
    Aggregate results from a completed MCTS tree.
    
    Args:
        root: The root node of the completed MCTS tree
        answer: The ground truth answer to validate against
    
    Returns:
        output: The predicted answer string
        correct: Whether prediction matches ground truth 
        reward: The aggregated reward
        confidence: Confidence score (reward/total_reward)
    """
    answer_dict = defaultdict(float)

    def visit(cur: MCTSNode) -> list[Tuple[Tuple[str, bool], int]]:
        # Skip unvisited or negative reward nodes
        if not cur.visits or cur.reward < 0:
            return []

        # For terminal nodes, check answer and add weighted reward
        if cur.is_terminal():
            if not cur.state or not cur.state.states:
                return []
            # Get answer from final state
            pred = extract_answer(cur.state.states[-1].subanswer)
            if pred is None:
                return []

            correct = abs(float(pred) - answer) < 1e-6

            # Calculate depth
            depth = 0
            node = cur
            while node.parent:
                depth += 1
                node = node.parent

            # Store tuple of (prediction string, correctness)
            key = (str(pred), correct)
            answer_dict[key] += cur.reward / depth
            return [(key, depth)]

        # Process children and their depths
        depth_dict = defaultdict(list)
        results = []
        for child in cur.children:
            child_results = visit(child)
            results.extend(child_results)
            for key, depth in child_results:
                depth_dict[key].append(depth)

        # Add weighted rewards from this node
        for key, depths in depth_dict.items():
            answer_dict[key] += cur.reward * len(depths) / sum(depths)

        return results

    # Traverse tree
    visit(root)

    if not answer_dict:
        return '', False, -10, 0

    # Sort by aggregated rewards
    answer_reward_list = sorted(answer_dict.items(), key=lambda x: x[1], reverse=True)
    (pred_str, is_correct), reward = answer_reward_list[0]

    # Calculate confidence as portion of total reward
    reward_sum = sum(x[1] for x in answer_reward_list)
    confidence = reward / reward_sum if reward_sum > 0 else 0

    return pred_str, is_correct, reward, confidence

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
