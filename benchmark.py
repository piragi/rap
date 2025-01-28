import json
import time
import os
from typing import Dict

import torch
from datasets import load_dataset
from tqdm import tqdm

from config import ModelParams
from main import Tokenizer, generate, load_model_params, load_weights
from mcts import mcts
from world_model import State
from aggregate import aggregate

def extract_answer(answer_text: str) -> float:
    """Extract numerical answer from the text."""
    try:
        if "The answer is" in answer_text:
            answer_str = answer_text.split("The answer is")[-1].strip().strip('.')
            # Remove any text after the number
            answer_str = ''.join(c for c in answer_str if c.isdigit() or c == '.' or c == '-')
            return float(answer_str)
    except:
        return None
    return None

def run_benchmark(dataset,
                  tokenizer: Tokenizer,
                  transformer_weights: dict,
                  model_params: ModelParams,
                  prefix: str,
                  n_samples: int = 100,
                  rollouts: int = 10,
                  depth_limit: int = 5,
                  confidence: int = 1,
                  action_generation: int = 5,
                  trace_file: str = "mcts_trace.txt",
                  start_idx: int = 0,
                  use_aggregate: bool = False) -> Dict:  # Added start_idx parameter
    """Run MCTS on GSM8K dataset and compute accuracy."""

    # Use subset of dataset if specified
    if n_samples > 0:  # Changed condition to check if n_samples is positive
        end_idx = min(start_idx + n_samples, len(dataset))
    else:
        end_idx = len(dataset)
    dataset = dataset.select(range(start_idx, end_idx))

    correct_mcts = 0
    correct_agg = 0
    total = 0
    results = []
    start_time_total = time.time()

    for idx, example in tqdm(enumerate(dataset, start=start_idx)):
        try:
            question = example['question']
            # Add error handling for target parsing
            try:
                target_str = example['answer'].split('####')[-1].strip()
                target = float(target_str)
            except (ValueError, IndexError) as e:
                print(f"\nError parsing target at index {idx}: {str(e)}")
                print(f"Problem answer string: {example['answer']}")
                continue

            # Write question and target to trace file before MCTS
            with open(trace_file, 'a') as f:
                print("\n" + "=" * 50, file=f)
                print(f"Index: {idx}", file=f)  # Added index logging
                print(f"Main Question: {question}", file=f)
                print(f"Target Answer: {target}", file=f)
                print("=" * 50 + "\n", file=f)

            # Initialize state with question
            init_state = State(states=[], prefix=prefix, question=question)

            # Run MCTS
            try:
                final_state, root = mcts(init_state, rollouts, depth_limit, action_generation, 
                                      tokenizer, transformer_weights, model_params, confidence)

                result_dict = {
                    'index': idx,
                    'question': question,
                    'target': target,
                }

                # Get MCTS result
                if final_state and final_state.states:
                    mcts_pred = extract_answer(final_state.states[-1].subanswer)
                    if mcts_pred is not None:
                        is_correct_mcts = abs(mcts_pred - target) < 1e-6
                        correct_mcts += int(is_correct_mcts)
                        result_dict.update({
                            'mcts_prediction': mcts_pred,
                            'mcts_correct': is_correct_mcts,
                            'mcts_steps': [state.subquestion + "\n" + state.subanswer 
                                         for state in final_state.states]
                        })

                # Get aggregation result only if flag is set
                if use_aggregate:
                    output, is_correct_agg, reward, conf = aggregate(root, target)
                    agg_pred = float(output) if output else None
                    if agg_pred is not None:
                        correct_agg += int(is_correct_agg)
                        result_dict.update({
                            'aggregate_prediction': agg_pred,
                            'aggregate_correct': is_correct_agg,
                            'aggregate_reward': reward,
                            'aggregate_confidence': conf
                        })

                results.append(result_dict)
                total += 1

                # Print appropriate progress info
                print(f"\nQuestion {idx}")
                print(f"MCTS - pred: {mcts_pred}, correct: {is_correct_mcts}")
                if use_aggregate:
                    print(f"Aggregation - pred: {agg_pred}, correct: {is_correct_agg}")
                print(f"Running accuracy - MCTS: {correct_mcts/total:.2%}", 
                      end="")
                if use_aggregate:
                    print(f", Aggregation: {correct_agg/total:.2%}")
                else:
                    print()

            except Exception as e:
                print(f"\nError in MCTS at index {idx}: {str(e)}")
                continue

        except Exception as e:
            print(f"\nUnexpected error at index {idx}: {str(e)}")
            continue

    total_time = time.time() - start_time_total
    # Prepare return dictionary
    results_dict = {
        'mcts_accuracy': correct_mcts / total if total > 0 else 0,
        'mcts_correct': correct_mcts,
        'total': total,
        'total_time_seconds': total_time,
        'results': results,
        'last_index': idx
    }
    
    # Add aggregation results only if flag is set
    if use_aggregate:
        results_dict.update({
            'aggregate_accuracy': correct_agg / total if total > 0 else 0,
            'aggregate_correct': correct_agg,
        })

    return results_dict

def run_cot_benchmark(dataset,
                      tokenizer: Tokenizer,
                      transformer_weights: dict,
                      model_params: ModelParams,
                      n_samples: int = None,
                      max_iterations: int = 10,
                      trace_file: str = "cot_trace.txt",
                      start_idx: int = 0) -> Dict:
    """
    Run chain of thought benchmark with multiple attempts per question
    Args:
        dataset: The dataset to evaluate on
        tokenizer: Tokenizer instance
        transformer_weights: Model weights
        model_params: Model parameters
        n_samples: Number of samples to evaluate (optional)
        max_iterations: Maximum number of attempts per question (default: 10)
        trace_file: File to write traces to
        start_idx: Starting index in dataset
    Returns:
        Dict containing accuracy and detailed results
    """
    prefix = """Q: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?\nA: Natalia sold 48 clips in April and half as many clips in May, so she sold 48 / 2 = 24 clips in May. Altogether, she sold 48 + 24 = 72 clips. The answer is 72.\n\nQ: Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?\nA: Since Weng earns $12 an hour for babysitting, she earns $12 / 60 = $0.2 per minute. Working 50 minutes, she earned $0.2 x 50 = $10. The answer is 10.\n\nQ: Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet?\nA: In the beginning, Betty has only half of the money she needs, which is 100 / 2 = $50. Her grandparents gave her twice as much as her parents, so they gave her 15 * 2 = $30. Now that she got $15 from her parents and $30 from her grandparents, she will need $100 - $15 - $30 = $55. Since she already has $50, she needs $55 - $50 = $5 more. The answer is 5.\n\nQ: Julie is reading a 120-page book. Yesterday, she was able to read 12 pages and today, she read twice as many pages as yesterday. If she wants to read half of the remaining pages tomorrow, how many pages should she read?\nA: Julie read twice as many pages as yesterday, so she read 12 * 2 = 24 pages today. Since yesterday, Julie read 12 + 24 = 36 pages. So, there are 120 - 36 = 84 pages left to be read. Since she wants to read half of the remaining pages, she should read 84 / 2 = 42 pages. The answer is 42.\n\n"""

    if n_samples:
        end_idx = min(start_idx + n_samples, len(dataset))
        dataset = dataset.select(range(start_idx, end_idx))
    else:
        dataset = dataset.select(range(start_idx, len(dataset)))

    correct = 0
    total = 0
    results = []

    for idx, example in tqdm(enumerate(dataset, start=start_idx)):
        try:
            question = example['question']
            # Add error handling for target parsing
            try:
                target_str = example['answer'].split('####')[-1].strip()
                target = float(target_str)
            except (ValueError, IndexError) as e:
                print(f"\nError parsing target at index {idx}: {str(e)}")
                print(f"Problem answer string: {example['answer']}")
                continue

            # Write question and target to trace file
            with open(trace_file, 'a') as f:
                print("\n" + "=" * 50, file=f)
                print(f"Index: {idx}", file=f)
                print(f"Question: {question}", file=f)
                print(f"Target Answer: {target}", file=f)
                print("=" * 50 + "\n", file=f)

            prompt = prefix + question + "\nA: "

            # Initialize variables for collecting predictions
            predictions = []
            all_attempts = []

            # Make all attempts regardless of correctness
            for attempt in range(max_iterations):
                try:
                    # Generate response using the model
                    tokens = tokenizer.encode(prompt, bos=False, eos=False, allowed_special="all")
                    tokens = torch.tensor([tokens]).cuda()

                    output_tokens = generate(transformer_weights, model_params, tokens, tokenizer, temperature=0.8, max_gen_len=2048)

                    response = tokenizer.decode(output_tokens[0].tolist())
                    reasoning = response.split(prompt)[-1].strip()

                    # Extract predicted answer
                    pred = extract_answer(reasoning)

                    if pred is not None:
                        predictions.append(pred)

                        attempt_result = {
                            'attempt_number': attempt + 1,
                            'reasoning': reasoning,
                            'predicted': pred,
                        }
                        all_attempts.append(attempt_result)

                        # Write trace to file
                        with open(trace_file, 'a') as f:
                            print(f"Attempt {attempt + 1}:", file=f)
                            print(f"Model reasoning:\n{reasoning}\n", file=f)
                            print(f"Predicted: {pred}\n", file=f)

                except Exception as e:
                    print(f"\nError in model generation at index {idx}, attempt {attempt + 1}: {str(e)}")
                    all_attempts.append({'attempt_number': attempt + 1, 'error': str(e)})
                    continue

            # Determine majority vote if we have any valid predictions
            if predictions:
                from collections import Counter
                prediction_counts = Counter(predictions)
                majority_pred = prediction_counts.most_common(1)[0][0]

                # Check if majority prediction is correct
                is_correct = abs(majority_pred - target) < 1e-6
                if is_correct:
                    correct += 1

                # Write majority vote results to trace file
                with open(trace_file, 'a') as f:
                    print("Majority Vote Results:", file=f)
                    print(f"All predictions: {predictions}", file=f)
                    print(f"Prediction counts: {dict(prediction_counts)}", file=f)
                    print(f"Selected prediction: {majority_pred}", file=f)
                    print(f"Correct: {is_correct}\n", file=f)
            else:
                is_correct = False
                majority_pred = None

            total += 1

            results.append({
                'index': idx,
                'question': question,
                'target': target,
                'attempts': all_attempts,
                'predictions': predictions,
                'majority_prediction': majority_pred,
                'final_correct': is_correct
            })

            print(f"\nQuestion {idx} - Running accuracy: {correct/total:.2%} ({correct}/{total})")
            print(f"Predictions: {predictions}")
            print(f"Majority vote: {majority_pred}, Success: {is_correct}")

        except Exception as e:
            print(f"\nUnexpected error at index {idx}: {str(e)}")
            continue

    accuracy = correct / total if total > 0 else 0

    return {'accuracy': accuracy, 'correct': correct, 'total': total, 'results': results, 'last_index': idx}

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2 or sys.argv[1] not in ['cot', 'rap']:
        print("Usage: python3 benchmark.py [cot|rap] [start_index] [--aggregate]")
        sys.exit(1)

    # Get start index and aggregate flag
    start_idx = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    use_aggregate = "--aggregate" in sys.argv

    # Load model components
    home_dir = os.path.expanduser("~")
    model_path = os.path.join(home_dir, ".llama", "checkpoints", "Llama3.2-3B")
    model_params = load_model_params(os.path.join(model_path, "params.json"))
    transformer_weights = load_weights(os.path.join(model_path, "consolidated.00.pth"))
    tokenizer = Tokenizer(model_path=os.path.join(model_path, "tokenizer.model"))

    # Load GSM8K dataset
    dataset = load_dataset("openai/gsm8k", "main")
    test_dataset = dataset['test']

    if sys.argv[1] == 'cot':
        print(f"Running Chain of Thought benchmark starting from index {start_idx}...")
        max_iterations = 3
        results = run_cot_benchmark(dataset=test_dataset,
                                    tokenizer=tokenizer,
                                    transformer_weights=transformer_weights,
                                    max_iterations=max_iterations,
                                    model_params=model_params,
                                    n_samples=100,
                                    start_idx=start_idx)
        output_file = f'gsm8k_cot_results_start:{start_idx}_iterations:{max_iterations}.json'
    else:  # rap
        print(f"Running Reasoning via Planning (RAP) benchmark starting from index {start_idx}...")
        prefix = json.load(open('prompts.json'))['repeated']['prompt']
        rollouts = 3
        results = run_benchmark(dataset=test_dataset,
                                tokenizer=tokenizer,
                                transformer_weights=transformer_weights,
                                model_params=model_params,
                                prefix=prefix,
                                n_samples=100,
                                rollouts=rollouts,
                                depth_limit=6,
                                confidence=1,
                                action_generation=3,
                                start_idx=start_idx)
        output_file = f'gsm8k_rap_results_start:{start_idx}_rollouts:{rollouts}.json'

    # Save results
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nFinal accuracy: {results['accuracy']:.2%}")
    print(f"Last processed index: {results['last_index']}")
