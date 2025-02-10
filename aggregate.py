from collections import defaultdict
from typing import Tuple

from mcts import MCTSNode, extract_answer

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
