import gc
import math
from typing import Optional

import torch
from torch import select_copy

from config import ModelParams
from tokenizer import Tokenizer
from weights import TransformerWeights
from world_model import Action, State, predict_action, predict_state

W_EXP = 1.

class MCTSNode:
    def __init__(self, state: Optional[State], action: Optional[Action], parent: Optional['MCTSNode'], reward: float):
        self.state = state
        self.action = action
        self.parent = parent
        self.visits = 0
        self.reward = reward
        self.children: list[MCTSNode] = []

    def is_terminal(self) -> bool:
        if self.state is not None and self.state.states:
            last_question = self.state.states[-1].subquestion
            if last_question.startswith("Now we can answer the question:"):
                return True
        return False

def select_node(node: MCTSNode) -> list[MCTSNode]:
    path = []
    while True:
        path.append(node)
        if node.children is None or len(node.children) == 0 or node.is_terminal():
            return path
        node = max(node.children, key=lambda child: child.reward + W_EXP * math.sqrt(math.log(max(1, node.visits)) / max(1, child.visits)))

def simulation(node: MCTSNode, depth_limit: int, action_generation: int, tokenizer: Tokenizer, transformer_weights: TransformerWeights,
               model_params: ModelParams) -> list[MCTSNode]:
    path = []
    current_node = node

    for _ in range(depth_limit):
        if current_node.state is None:
            current_node.state = predict_state(current_node.parent.state, current_node.action, tokenizer, transformer_weights, model_params)
        if current_node.is_terminal():
            path.append(current_node)
            break
        if not current_node.children:
            #TODO: parallelization potential through using kvcache and batched state
            for _ in range(action_generation):
                action, fast_reward = predict_action(current_node.state, tokenizer, transformer_weights, model_params)
                current_node.children.append(MCTSNode(state=None, action=action, parent=current_node, reward=fast_reward))

        current_node = max(current_node.children, key=lambda child: child.reward)
        path.append(current_node)
    return path

def backpropagation(path: list[MCTSNode]) -> float:
    cum_reward = 0.0
    for node in path:
        cum_reward += node.reward
        node.visits += 1
    return cum_reward

def get_robust_path(root: MCTSNode, c: float = 1.0) -> list[MCTSNode]:
    path = []
    current = root

    while current.children:
        current = max(current.children, key=lambda n: n.reward + c * math.sqrt(math.log(max(1, current.visits)) / max(1, n.visits)))
        path.append(current)
        if current.is_terminal():
            break
    return path

def mcts(init_state: State, rollouts: int, depth_limit: int, action_generation: int, tokenizer: Tokenizer, transformer_weights: TransformerWeights,
         model_params: ModelParams) -> State:
    root = MCTSNode(init_state, None, None, 0)
    for _ in range(rollouts):
        path = select_node(root)
        last_node = path[-1]
        simulation_path = simulation(last_node, depth_limit, action_generation, tokenizer, transformer_weights, model_params)
        backpropagation(simulation_path)

    # Select best path after all iterations
    best_path = get_robust_path(root)

    # Return final state from best path
    if best_path:
        return best_path[-1].state
    return init_state
