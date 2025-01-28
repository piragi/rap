from collections import defaultdict
from typing import Tuple

def aggregate(root: MCTSNode, answer: str) -> Tuple[str, bool, float, float]:
    """
    Aggregate results from a completed MCTS tree.
    Args:
        root: The root node of the completed MCTS tree
        answer: The ground truth answer to check against
    Returns:
        output: The predicted answer
        correct: Whether prediction matches ground truth 
        reward: The aggregated reward
        confidence: Confidence score (reward/total_reward)
    """
    answer_dict = defaultdict(float)

    def visit(cur: MCTSNode) -> list[Tuple[bool, int]]:
        # Skip unvisited or negative reward nodes
        if not cur.visits or cur.reward < 0:
            return []

        # For terminal nodes, check answer and add weighted reward
        if cur.is_terminal():
            if not cur.state or not cur.state.states:
                return []
            # Get answer from final state
            pred = cur.state.states[-1].subanswer
            correct = judge_answer_gsm8k(pred, answer)  # You'll need this function
            # Weight reward by depth (use parent links to calculate depth)
            depth = 0
            node = cur
            while node.parent:
                depth += 1
                node = node.parent
            answer_dict[correct] += cur.reward / depth
            return [(correct, depth)]

        # Process children and their depths
        depth_dict = defaultdict(list)
        results = []
        for child in cur.children:
            child_results = visit(child)
            results.extend(child_results)
            for correct, depth in child_results:
                depth_dict[correct].append(depth)

        # Add weighted rewards from this node
        for correct, depths in depth_dict.items():
            answer_dict[correct] += cur.reward * len(depths) / sum(depths)

        return results

    # Traverse tree
    visit(root)

    if not answer_dict:
        return '', False, -10, 0

    # Sort by aggregated rewards
    answer_reward_list = sorted(answer_dict.items(), key=lambda x: x[1], reverse=True) 
    (output, correct), reward = answer_reward_list[0]
    
    # Calculate confidence as portion of total reward
    reward_sum = sum(x[1] for x in answer_reward_list)
    confidence = reward / reward_sum if reward_sum > 0 else 0

    return output, correct, reward, confidence
