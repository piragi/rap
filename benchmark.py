import json
import os
import time
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

from datasets import load_dataset
from tqdm import tqdm

from aggregate import aggregate
from config import ModelParams, load_model_params
from main import generate, prepare_tokens
from mcts import mcts
from token_tracker import TokenUsageStats
from tokenizer import Tokenizer
from weights import load_weights
from world_model import State

@dataclass
class BenchmarkResult:
    """Dataclass to store benchmark results"""
    accuracy: float
    correct: int
    total: int
    results: List[Dict]
    last_index: int
    total_time_seconds: Optional[float] = None
    aggregate_accuracy: Optional[float] = None
    aggregate_correct: Optional[int] = None
    token_stats: Optional[Dict] = None

def extract_answer(answer_text: str) -> Optional[float]:
    """Extract numerical answer from text."""
    try:
        if "The answer is" in answer_text:
            answer_str = answer_text.split("The answer is")[-1].strip().rstrip('.')
            answer_str = ''.join(c for c in answer_str if c.isdigit() or c == '.' or c == '-')
            answer_str = answer_str.rstrip('.')
            print(answer_str)
            return float(answer_str)
    except:
        pass
    return None

def parse_target(example: Dict) -> Optional[float]:
    """Parse target answer from example."""
    try:
        target_str = example['answer'].split('####')[-1].strip()
        return float(target_str)
    except (ValueError, IndexError):
        return None

def write_trace(trace_file: str, idx: int, question: str, target: float, **kwargs):
    """Write trace information to file."""
    with open(trace_file, 'a') as f:
        print("\n" + "=" * 50, file=f)
        print(f"Index: {idx}", file=f)
        print(f"Question: {question}", file=f)
        print(f"Target Answer: {target}", file=f)
        for key, value in kwargs.items():
            print(f"{key}: {value}", file=f)
        print("=" * 50 + "\n", file=f)

class ModelRunner:
    """Class to handle model interactions"""
    def __init__(self, tokenizer: Tokenizer, weights: dict, params: ModelParams):
        self.tokenizer = tokenizer
        self.weights = weights
        self.params = params

    def generate_response(self,
                          prompts: Union[str, List[str]],
                          temperature: float = 0.9,
                          token_stats: Optional[TokenUsageStats] = None,
                          track_method: Optional[str] = None) -> Union[str, List[str]]:
        """Generate responses for one or multiple prompts"""
        if isinstance(prompts, str):
            prompts = [prompts]
            is_single = True
        else:
            is_single = False

        # Use prepare_tokens helper for consistent batching
        batched_tokens = prepare_tokens(prompts, self.tokenizer)
        input_length = batched_tokens.size(1)

        output_tokens = generate(self.weights,
                                 self.params,
                                 batched_tokens,
                                 self.tokenizer,
                                 temperature=temperature,
                                 max_gen_len=input_length + 200,
                                 token_stats=token_stats,
                                 track_method=track_method)

        # Decode complete sequences
        outputs = []
        for tokens in output_tokens:
            text = self.tokenizer.decode(tokens.tolist())
            outputs.append(text)

        return outputs[0] if is_single else outputs

class BenchmarkRunner:
    """Base class for running benchmarks"""
    def __init__(self, dataset, model_runner: ModelRunner, start_idx: int = 0):
        self.dataset = dataset
        self.model_runner = model_runner
        self.start_idx = start_idx
        self.token_stats = TokenUsageStats()

    def prepare_dataset(self, n_samples: Optional[int]) -> None:
        if n_samples and n_samples > 0:
            end_idx = min(self.start_idx + n_samples, len(self.dataset))
            self.dataset = self.dataset.select(range(self.start_idx, end_idx))
        else:
            self.dataset = self.dataset.select(range(self.start_idx, len(self.dataset)))

class MCTSBenchmark(BenchmarkRunner):
    """MCTS-specific benchmark implementation"""
    def run(self,
            prefix: str,
            rollouts: int = 10,
            depth_limit: int = 5,
            confidence: int = 1,
            action_generation: int = 5,
            trace_file: str = "mcts_trace.txt",
            use_aggregate: bool = False) -> BenchmarkResult:

        correct_mcts = correct_agg = total = 0
        results = []
        self.token_stats = TokenUsageStats()
        start_time = time.time()

        for idx, example in tqdm(enumerate(self.dataset, start=self.start_idx)):
            self.token_stats.start_new_run()
            target = parse_target(example)
            if target is None:
                continue

            write_trace(trace_file, idx, example['question'], target)

            try:
                result = self._process_example(example, target, prefix, rollouts, depth_limit, action_generation, confidence, use_aggregate)

                if result:
                    results.append(result)
                    correct_mcts += int(result.get('mcts_correct', False))
                    correct_agg += int(result.get('aggregate_correct', False))
                    total += 1

                    print(f"\nQuestion {idx}")
                    print(f"Target: {target}")
                    print(f"MCTS - pred: {result.get('mcts_prediction')}, correct: {result.get('mcts_correct')}")
                    if use_aggregate:
                        print(f"Aggregation - pred: {result.get('aggregate_prediction')}, correct: {result.get('aggregate_correct')}")
                    print(f"Running accuracy - MCTS: {correct_mcts/total:.2%}", end="")
                    if use_aggregate:
                        print(f", Aggregation: {correct_agg/total:.2%}")
                    else:
                        print()

            except Exception as e:
                print(f"\nDetailed error for example {idx}:")
                print(f"  Question: {example['question'][:100]}...")
                print(f"  Error type: {type(e).__name__}")
                print(f"  Error message: {str(e)}")
                print(f"  Stack trace:")
                import traceback
                traceback.print_exc()
                continue

        return BenchmarkResult(accuracy=correct_mcts / total if total > 0 else 0,
                               correct=correct_mcts,
                               total=total,
                               results=results,
                               last_index=idx,
                               total_time_seconds=time.time() - start_time,
                               aggregate_accuracy=correct_agg / total if use_aggregate and total > 0 else None,
                               aggregate_correct=correct_agg if use_aggregate else None,
                               token_stats=self.token_stats.get_stats())

    def _process_example(self, example: Dict, target: float, prefix: str, rollouts: int, depth_limit: int, action_generation: int, confidence: int,
                         use_aggregate: bool) -> Optional[Dict]:
        init_state = State(states=[], prefix=prefix, question=example['question'])
        final_state, root = mcts(init_state,
                                 rollouts,
                                 depth_limit,
                                 action_generation,
                                 self.model_runner.tokenizer,
                                 self.model_runner.weights,
                                 self.model_runner.params,
                                 confidence,
                                 token_stats=self.token_stats)

        if not final_state or not final_state.states:
            self.token_stats.discard_run()  # Discard tokens from failed run
            return None

        self.token_stats.commit_run()

        result = {
            'index': example.get('id', -1),
            'question': example['question'],
            'target': target,
        }

        # Process MCTS results
        mcts_pred = extract_answer(final_state.states[-1].subanswer)
        if mcts_pred is not None:
            result.update({
                'mcts_prediction': mcts_pred,
                'mcts_correct': abs(mcts_pred - target) < 1e-6,
                'mcts_steps': [f"{state.subquestion}\n{state.subanswer}" for state in final_state.states]
            })

        # Process aggregate results if requested
        if use_aggregate:
            output, is_correct, reward, conf = aggregate(root, target)
            if output:
                result.update({
                    'aggregate_prediction': float(output),
                    'aggregate_correct': is_correct,
                    'aggregate_reward': reward,
                    'aggregate_confidence': conf
                })

        return result

class CoTBenchmark(BenchmarkRunner):
    """Chain of Thought benchmark implementation with batched iterations"""
    def run(self, max_iterations: int = 10, batch_size: int = 1, trace_file: str = "cot_trace.txt") -> BenchmarkResult:
        correct = total = 0
        results = []
        start_time = time.time()

        # Standard CoT prefix with examples
        prefix = """Q: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?\nA: Natalia sold 48 clips in April and half as many clips in May, so she sold 48 / 2 = 24 clips in May. Altogether, she sold 48 + 24 = 72 clips. The answer is 72.\n\nQ: Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?\nA: Since Weng earns $12 an hour for babysitting, she earns $12 / 60 = $0.2 per minute. Working 50 minutes, she earned $0.2 x 50 = $10. The answer is 10.\n\nQ: Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet?\nA: In the beginning, Betty has only half of the money she needs, which is 100 / 2 = $50. Her grandparents gave her twice as much as her parents, so they gave her 15 * 2 = $30. Now that she got $15 from her parents and $30 from her grandparents, she will need $100 - $15 - $30 = $55. Since she already has $50, she needs $55 - $50 = $5 more. The answer is 5.\n\nQ: Julie is reading a 120-page book. Yesterday, she was able to read 12 pages and today, she read twice as many pages as yesterday. If she wants to read half of the remaining pages tomorrow, how many pages should she read?\nA: Julie read twice as many pages as yesterday, so she read 12 * 2 = 24 pages today. Since yesterday, Julie read 12 + 24 = 36 pages. So, there are 120 - 36 = 84 pages left to be read. Since she wants to read half of the remaining pages, she should read 84 / 2 = 42 pages. The answer is 42.\n\n"""

        for idx, example in tqdm(enumerate(self.dataset, start=self.start_idx)):
            self.token_stats.start_new_run()
            target = parse_target(example)
            if target is None:
                continue

            write_trace(trace_file, idx, example['question'], target)

            try:
                result = self._process_example(example, target, prefix, max_iterations, batch_size, trace_file)

                if result:
                    results.append(result)
                    correct += int(result['final_correct'])
                    total += 1

                    print(f"\nQuestion {idx} - Running accuracy: {correct/total:.2%} ({correct}/{total})")
                    print(f"Predictions: {result['predictions']}")
                    print(f"Majority vote: {result['majority_prediction']}, Success: {result['final_correct']}")

                self.token_stats.commit_run()

            except Exception as e:
                print(f"\nDetailed error for example {idx}:")
                print(f"  Question: {example['question'][:100]}...")
                print(f"  Error type: {type(e).__name__}")
                print(f"  Error message: {str(e)}")
                print(f"  Stack trace:")
                import traceback
                traceback.print_exc()
                self.token_stats.discard_run()
                continue

        return BenchmarkResult(accuracy=correct / total if total > 0 else 0,
                               correct=correct,
                               total=total,
                               results=results,
                               last_index=idx,
                               total_time_seconds=time.time() - start_time,
                               token_stats=self.token_stats.get_stats())

    def _process_example(self, example: Dict, target: float, prefix: str, max_iterations: int, batch_size: int, trace_file: str) -> Optional[Dict]:
        """Process a single example with batched attempts"""
        prompt = prefix + example['question'] + "\nA: "
        predictions = []
        all_attempts = []

        # Process iterations in batches
        for i in range(0, max_iterations, batch_size):
            curr_batch_size = min(batch_size, max_iterations - i)
            try:
                # Generate batch_size responses at once
                reasonings = self.model_runner.generate_response([prompt] * curr_batch_size,
                                                                 temperature=0.8,
                                                                 token_stats=self.token_stats,
                                                                 track_method='cot')

                # Process each response in batch
                for j, reasoning in enumerate(reasonings):
                    attempt = i + j + 1
                    reasoning = reasoning.split(prompt)[-1].strip()
                    pred = extract_answer(reasoning)

                    if pred is not None:
                        predictions.append(pred)
                        attempt_result = {
                            'attempt_number': attempt,
                            'reasoning': reasoning,
                            'predicted': pred,
                        }
                        all_attempts.append(attempt_result)

                        # Write trace
                        with open(trace_file, 'a') as f:
                            print(f"Attempt {attempt}:", file=f)
                            print(f"Model reasoning:\n{reasoning}\n", file=f)
                            print(f"Predicted: {pred}\n", file=f)

            except Exception as e:
                print(f"\nError in model generation, batch starting at attempt {i + 1}: {str(e)}")
                for j in range(curr_batch_size):
                    all_attempts.append({'attempt_number': i + j + 1, 'error': str(e)})
                continue

        # Determine majority vote if we have any valid predictions
        if predictions:
            prediction_counts = Counter(predictions)
            majority_pred = prediction_counts.most_common(1)[0][0]
            is_correct = abs(majority_pred - target) < 1e-6

            # Write majority vote results to trace
            with open(trace_file, 'a') as f:
                print("Majority Vote Results:", file=f)
                print(f"All predictions: {predictions}", file=f)
                print(f"Prediction counts: {dict(prediction_counts)}", file=f)
                print(f"Selected prediction: {majority_pred}", file=f)
                print(f"Correct: {is_correct}\n", file=f)
        else:
            is_correct = False
            majority_pred = None

        return {
            'index': example.get('id', -1),
            'question': example['question'],
            'target': target,
            'attempts': all_attempts,
            'predictions': predictions,
            'majority_prediction': majority_pred,
            'final_correct': is_correct
        }

def main():
    # Parse command line arguments
    if len(sys.argv) < 2 or sys.argv[1] not in ['cot', 'rap']:
        print("Usage: python3 benchmark.py [cot|rap] [start_index] [--aggregate]")
        sys.exit(1)

    benchmark_type = sys.argv[1]
    start_idx = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    use_aggregate = "--aggregate" in sys.argv

    n_samples = 1000
    # Initialize model
    home_dir = os.path.expanduser("~")
    model_path = os.path.join(home_dir, ".llama", "checkpoints", "Llama3.2-3B")
    model_runner = ModelRunner(tokenizer=Tokenizer(model_path=os.path.join(model_path, "tokenizer.model")),
                               weights=load_weights(os.path.join(model_path, "consolidated.00.pth")),
                               params=load_model_params(os.path.join(model_path, "params.json")))

    # Load dataset
    dataset = load_dataset("openai/gsm8k", "main")['test']

    # Initialize appropriate benchmark
    if benchmark_type == 'cot':
        print(f"Running Chain of Thought benchmark...")
        print(f"Start index: {start_idx}")
        print(f"Number of samples: {n_samples}")

        benchmark = CoTBenchmark(dataset, model_runner, start_idx)
        benchmark.prepare_dataset(n_samples=n_samples)

        max_iteration = 5
        results = benchmark.run(max_iterations=max_iteration, trace_file=f"cot_trace_{start_idx}.txt", batch_size=7)
        output_file = f'gsm8k_cot_iterations{max_iteration}_results_start{start_idx}_samples{n_samples}.json'

    else:  # rap
        print(f"Running Reasoning via Planning (RAP) benchmark...")
        print(f"Start index: {start_idx}")
        print(f"Number of samples: {n_samples}")
        print(f"Aggregation enabled: {use_aggregate}")

        benchmark = MCTSBenchmark(dataset, model_runner, start_idx)
        benchmark.prepare_dataset(n_samples=n_samples)

        # Load RAP-specific parameters
        prefix = json.load(open('prompts.json'))['repeated']['prompt']

        results = benchmark.run(prefix=prefix,
                                rollouts=1,
                                depth_limit=6,
                                confidence=1,
                                action_generation=1,
                                trace_file=f"rap_trace_{start_idx}.txt",
                                use_aggregate=use_aggregate)
        output_file = f'gsm8k_rap_results_start{start_idx}_samples{n_samples}.json'

    # Save results
    with open(output_file, 'w') as f:
        json.dump(vars(results), f, indent=2)

    # Print summary
    print("\nBenchmark Summary:")
    print(f"Type: {benchmark_type.upper()}")
    print(f"Samples processed: {results.total}")
    print(f"Final accuracy: {results.accuracy:.2%}")
    if benchmark_type == 'rap' and use_aggregate:
        print(f"Aggregation accuracy: {results.aggregate_accuracy:.2%}")
    print(f"Last processed index: {results.last_index}")
    print(f"Results saved to: {output_file}")

if __name__ == "__main__":
    import sys
    main()
