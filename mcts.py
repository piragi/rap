import math
from typing import Optional

import numpy as np

from config import ModelParams
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

    def is_terminal(self) -> bool:
        if self.state is not None and self.state.states:
            last_question = self.state.states[-1].subquestion.strip()
            if last_question.startswith("Now we can answer the question:"):
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
        node.visits += 1
        path.append(node)
        if node.children is None or len(node.children) == 0 or node.is_terminal():
            return path
        node = max(node.children, key=lambda child: child.q_value + W_EXP * np.sqrt(np.log(node.visits) / max(1, child.visits)))

def simulation(node: MCTSNode, depth_limit: int, action_generation: int, tokenizer: Tokenizer, transformer_weights: TransformerWeights,
               model_params: ModelParams) -> list[MCTSNode]:
    path = []
    current_node = node

    for _ in range(depth_limit):
        if current_node.state is None:
            current_node.state = predict_state(current_node.parent.state, current_node.action, tokenizer, transformer_weights, model_params)
            current_node.reward = calculate_reward(current_node.fast_reward, current_node.state.states[-1].confidence)
        if current_node.is_terminal():
            path.append(current_node)
            break

        if not current_node.children:
            # Generate all actions at once
            actions_with_rewards = predict_action(current_node.state, tokenizer, transformer_weights, model_params, action_generation)
            # Create children nodes
            for action, fast_reward in actions_with_rewards:
                current_node.children.append(MCTSNode(state=None, action=action, parent=current_node, reward=0., fast_reward=fast_reward))

        current_node = max(current_node.children, key=lambda child: child.fast_reward)
        path.append(current_node)
    return path

def backpropagation(path: list[MCTSNode]) -> float:
    # For each node in path, collect sequence of future rewards
    for i, node in enumerate(reversed(path)):
        future_rewards = [n.reward for n in path[len(path) - i - 1:]]
        node.cum_rewards.append(future_rewards)
    return sum(node.reward for node in path)

def get_highest_reward_path(root: MCTSNode) -> tuple[float, list[MCTSNode]]:
    """
    Find path with highest reward through DFS.
    Returns (total_reward, path) tuple.
    """
    def dfs(path: list[MCTSNode]) -> tuple[float, list[MCTSNode]]:
        current = path[-1]
        if current.is_terminal():
            return sum(node.reward for node in path[1:]), path

        if current.children is None:
            return float('-inf'), path

        visited_children = [c for c in current.children if c.state is not None]
        if not visited_children:
            return float('-inf'), path

        return max((dfs(path + [child]) for child in visited_children), key=lambda x: x[0])

    return dfs([root])

def mcts(init_state: State, rollouts: int, depth_limit: int, action_generation: int, tokenizer: Tokenizer, transformer_weights: TransformerWeights,
         model_params: ModelParams) -> State:
    root = MCTSNode(init_state, init_state.question, None, 0., 0.)
    for _ in range(rollouts):
        path = select_node(root)
        last_node = path[-1]
        simulation_path = simulation(last_node, depth_limit - len(path), action_generation, tokenizer, transformer_weights, model_params)
        path.extend(simulation_path)
        backpropagation(path)

    reward, best_path = get_highest_reward_path(root)
    print_mcts(root)
    if reward == float('-inf'):
        print("No valid complete path found")
        return None
    print_reasoning_trace(best_path)
    return best_path[-1].state

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
