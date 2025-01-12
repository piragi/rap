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

    prefix = """Given a question, please decompose it into sub-questions. For each sub-question, please answer it in a complete sentence, ending with "The answer is". When the original question is answerable, please start the subquestion with "Now we can answer the question: ".

Here are some examples: Question 1: Four years ago, Kody was only half as old as Mohamed. If Mohamed is currently twice as 30 years old, how old is Kody?
Question 1.1: How old is Mohamed?
Answer 1.1: He is currently 30 * 2 = 60 years old. The answer is 60.
Question 1.2: How old was Mohamed four years ago?
Answer 1.2: Four years ago, he must have been 60 - 4 = 56 years old. The answer is 56.
Question 1.3: How old was Kody four years ago?
Answer 1.3: Kody was half as old as Mohamed four years ago. Thus, Kody was 56 / 2 = 28 years old. The answer is 28.
Question 1.4: Now we can answer the question: How old is Kody?
Answer 1.4: She is currently 28 + 4 = 32 years old. The answer is 32.

Question 2: On a moonless night, three fireflies danced in the evening breeze. They were joined by four less than a dozen more fireflies before two of the fireflies flew away. How many fireflies remained?
Question 2.1: How many fireflies joined?
Answer 2.1: The fireflies were joined by four less than a dozen more fireflies, which are 12 - 4 = 8 fireflies. The answer is 8.
Question 2.2: Now we can answer the question: How many fireflies remained?
Answer 2.2: Three fireflies were dancing originally. They were joined by 8 fireflies before two of them flew away. So there were 3 + 8 - 2 = 9 remaining. The answer is 9.

Question 3: Ali has four $10 bills and six $20 bills that he saved after working for Mr. James on his farm. Ali gives her sister half of the total money he has and uses 3/5 of the remaining amount of money to buy dinner. Calculate the amount of money he has after buying the dinner.
Question 3.1: How much money does Ali have in total?
Answer 3.1: Ali has four $10 bills and six $20 bills. So he has 4 * 10 + 6 * 20 = 160 dollars. The answer is 160.
Question 3.2: How much money does Ali give to his sister?
Answer 3.2: Ali gives half of the total money he has to his sister. So he gives 160 / 2 = 80 dollars to his sister. The answer is 80.
Question 3.3: How much money does Ali have after giving his sister the money?
Answer 3.3: After giving his sister the money, Ali has 160 - 80 = 80 dollars left. The answer is 80.
Question 3.4: How much money does Ali use to buy dinner?
Answer 3.4: Ali uses 3/5 of the remaining amount of money to buy dinner. So he uses 80 * 3/5 = 48 dollars to buy dinner. The answer is 48.
Question 3.5: Now we can answer the question: How much money does Ali have after buying the dinner?
Answer 3.5: After buying the dinner, Ali has 80 - 48 = 32 dollars left. The answer is 32.

Question 4: A car is driving through a tunnel with many turns. After a while, the car must travel through a ring that requires a total of 4 right-hand turns. After the 1st turn, it travels 5 meters. After the 2nd turn, it travels 8 meters. After the 3rd turn, it travels a little further and at the 4th turn, it immediately exits the tunnel. If the car has driven a total of 23 meters around the ring, how far did it have to travel after the 3rd turn?
Question 4.1: How far did the car travel except for the 3rd turn?
Answer 4.1: It travels 5 meters after the 1st, 8 meters after the 2nd, and 0 meters after the 4th turn. It’s a total of 5 + 8 + 0 = 13 meters. The answer is 13.
Question 4.2: Now we can answer the question: How far did the car have to travel after the 3rd turn?
Answer 4.2: The car has driven a total of 23 meters around the ring. It travels 13 meters except for the 3rd turn. So it has to travel 23 - 13 = 10 meters after the 3rd turn. The answer is 10."""

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
    prefix = """Q: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?\nA: Natalia sold 48 clips in April and half as many clips in May, so she sold 48 / 2 = 24 clips in May. Altogether, she sold 48 + 24 = 72 clips. The answer is 72.\n\n
Q: Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?\nA: Since Weng earns $12 an hour for babysitting, she earns $12 / 60 = $0.2 per minute. Working 50 minutes, she earned $0.2 x 50 = $10. The answer is 10.\n\n
Q: Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet?\nA: In the beginning, Betty has only half of the money she needs, which is 100 / 2 = $50. Her grandparents gave her twice as much as her parents, so they gave her 15 * 2 = $30. Now that she got $15 from her parents and $30 from her grandparents, she will need $100 - $15 - $30 = $55. Since she already has $50, she needs $55 - $50 = $5 more. The answer is 5.\n\n
Q: Julie is reading a 120-page book. Yesterday, she was able to read 12 pages and today, she read twice as many pages as yesterday. If she wants to read half of the remaining pages tomorrow, how many pages should she read?\nA: Julie read twice as many pages as yesterday, so she read 12 * 2 = 24 pages today. Since yesterday, Julie read 12 + 24 = 36 pages. So, there are 120 - 36 = 84 pages left to be read. Since she wants to read half of the remaining pages, she should read 84 / 2 = 42 pages. The answer is 42.\n\n"""

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

                    output_tokens = generate(transformer_weights, model_params, tokens, tokenizer, temperature=0.8, max_gen_len=512)

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
        rollouts = 3
        results = run_benchmark(dataset=test_dataset,
                                tokenizer=tokenizer,
                                transformer_weights=transformer_weights,
                                model_params=model_params,
                                n_samples=100,
                                rollouts=rollouts,
                                depth_limit=6,
                                action_generation=4,
                                start_idx=start_idx)
        output_file = f'gsm8k_rap_results_start:{start_idx}_rollouts:{rollouts}.json'

    # Save results
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nFinal accuracy: {results['accuracy']:.2%}")
    print(f"Last processed index: {results['last_index']}")
