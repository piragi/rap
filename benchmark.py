import json
import os
from typing import Dict

import torch
from datasets import load_dataset
from tqdm import tqdm

from config import ModelParams
from main import Tokenizer, generate, load_model_params, load_weights
from mcts import mcts
from world_model import State


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
                  n_samples: int = None,
                  rollouts: int = 10,
                  depth_limit: int = 5,
                  action_generation: int = 5,
                  trace_file: str = "mcts_trace.txt",
                  start_idx: int = 0) -> Dict:  # Added start_idx parameter
    """Run MCTS on GSM8K dataset and compute accuracy."""

    # Use subset of dataset if specified
    if n_samples > 0:  # Changed condition to check if n_samples is positive
        end_idx = min(start_idx + n_samples, len(dataset))
    else:
        end_idx = len(dataset)
    dataset = dataset.select(range(start_idx, end_idx))

    correct = 0
    total = 0
    results = []

    prefix = """Given a question, please decompose it into sub-questions..."""  # Your existing prefix

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
            init_state = State(states=[], prefix=prefix, question="Question 5: " + question)

            # Run MCTS
            try:
                final_state = mcts(init_state, rollouts, depth_limit, action_generation, tokenizer, transformer_weights, model_params)

                # Extract predicted answer
                if final_state and final_state.states:
                    pred = extract_answer(final_state.states[-1].subanswer)
                    if pred is not None:
                        # Compare prediction with target
                        is_correct = abs(pred - target) < 1e-6
                        correct += int(is_correct)

                        results.append({
                            'index': idx,  # Added index to results
                            'question': question,
                            'target': target,
                            'predicted': pred,
                            'correct': is_correct,
                            'steps': [state.subquestion + "\n" + state.subanswer for state in final_state.states]
                        })
            except Exception as e:
                print(f"\nError in MCTS at index {idx}: {str(e)}")
                continue

            total += 1

            # Print running accuracy with index information
            print(f"\nQuestion {idx} - Running accuracy: {correct/total:.2%} ({correct}/{total})")

        except Exception as e:
            print(f"\nUnexpected error at index {idx}: {str(e)}")
            continue

    accuracy = correct / total if total > 0 else 0

    return {'accuracy': accuracy, 'correct': correct, 'total': total, 'results': results, 'last_index': idx}

def run_cot_benchmark(dataset,
                     tokenizer: Tokenizer,
                     transformer_weights: dict,
                     model_params: ModelParams,
                     n_samples: int = None,
                     trace_file: str = "cot_trace.txt",
                     start_idx: int = 0) -> Dict:  # Added start_idx parameter
    prefix = """Q: Natalia sold clips..."""  # Your existing prefix

    # Use subset of dataset if specified
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
                print(f"Index: {idx}", file=f)  # Added index logging
                print(f"Question: {question}", file=f)
                print(f"Target Answer: {target}", file=f)
                print("=" * 50 + "\n", file=f)

            prompt = prefix + question + "\nA: "
            
            try:
                # Generate response using the model
                tokens = tokenizer.encode(prompt, bos=False, eos=False, allowed_special="all")
                tokens = torch.tensor([tokens]).cuda()
                
                output_tokens = generate(
                    transformer_weights,
                    model_params,
                    tokens,
                    tokenizer,
                    temperature=0.9,
                    max_gen_len=512
                )
                
                response = tokenizer.decode(output_tokens[0].tolist())
                reasoning = response.split(prompt)[-1].strip()

                # Extract predicted answer
                pred = extract_answer(reasoning)
                
                if pred is not None:
                    # Compare prediction with target
                    is_correct = abs(pred - target) < 1e-6
                    correct += int(is_correct)
                    
                    results.append({
                        'index': idx,  # Added index to results
                        'question': question,
                        'target': target,
                        'predicted': pred,
                        'correct': is_correct,
                        'steps': [reasoning]
                    })

                    # Write trace to file
                    with open(trace_file, 'a') as f:
                        print(f"Model reasoning:\n{reasoning}\n", file=f)
                        print(f"Predicted: {pred}", file=f)
                        print(f"Correct: {is_correct}\n", file=f)

            except Exception as e:
                print(f"\nError in model generation at index {idx}: {str(e)}")
                continue

            total += 1
            print(f"\nQuestion {idx} - Running accuracy: {correct/total:.2%} ({correct}/{total})")

        except Exception as e:
            print(f"\nUnexpected error at index {idx}: {str(e)}")
            continue

    accuracy = correct / total if total > 0 else 0

    return {
        'accuracy': accuracy,
        'correct': correct,
        'total': total,
        'results': results,
        'last_index': idx
    }

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2 or sys.argv[1] not in ['cot', 'rap']:
        print("Usage: python3 benchmark.py [cot|rap] [start_index]")
        sys.exit(1)

    # Get start index if provided
    start_idx = int(sys.argv[2]) if len(sys.argv) > 2 else 0

    # Load model components
    home_dir = os.path.expanduser("~")
    model_path = os.path.join(home_dir, ".llama", "checkpoints", "Llama3.2-3B-Instruct")
    model_params = load_model_params(os.path.join(model_path, "params.json"))
    transformer_weights = load_weights(os.path.join(model_path, "consolidated.00.pth"))
    tokenizer = Tokenizer(model_path=os.path.join(model_path, "tokenizer.model"))

    # Load GSM8K dataset
    dataset = load_dataset("openai/gsm8k", "main")
    test_dataset = dataset['test']

    if sys.argv[1] == 'cot':
        print(f"Running Chain of Thought benchmark starting from index {start_idx}...")
        results = run_cot_benchmark(
            dataset=test_dataset,
            tokenizer=tokenizer,
            transformer_weights=transformer_weights,
            model_params=model_params,
            n_samples=100,
            start_idx=start_idx)
        output_file = f'gsm8k_cot_results_{start_idx}.json'
    else:  # rap
        print(f"Running Reasoning via Planning (RAP) benchmark starting from index {start_idx}...")
        results = run_benchmark(
            dataset=test_dataset,
            tokenizer=tokenizer,
            transformer_weights=transformer_weights,
            model_params=model_params,
            n_samples=100,
            rollouts=6,
            depth_limit=6,
            action_generation=4,
            start_idx=start_idx)
        output_file = f'gsm8k_rap_results_{start_idx}.json'

    # Save results
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nFinal accuracy: {results['accuracy']:.2%}")
    print(f"Last processed index: {results['last_index']}")
