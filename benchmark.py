import json
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
from datasets import load_dataset
from tqdm import tqdm

from config import ModelParams, load_model_params
from main import generate
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
            answer_str = answer_text.split("The answer is")[-1].strip().strip('.')
            answer_str = ''.join(c for c in answer_str if c.isdigit() or c == '.' or c == '-')
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

    def generate_response(self, prompt: str, temperature: float = 0.9) -> str:
        tokens = self.tokenizer.encode(prompt, bos=False, eos=False, allowed_special="all")
        tokens = torch.tensor([tokens]).cuda()
        output_tokens = generate(self.weights, self.params, tokens, self.tokenizer, temperature=temperature, max_gen_len=2048)
        return self.tokenizer.decode(output_tokens[0].tolist())

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

def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ['cot', 'rap']:
        print("Usage: python3 benchmark.py [cot|rap] [start_index] [--aggregate]")
        sys.exit(1)

    start_idx = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    use_aggregate = "--aggregate" in sys.argv

    # Initialize model
    home_dir = os.path.expanduser("~")
    model_path = os.path.join(home_dir, ".llama", "checkpoints", "Llama3.2-3B")
    model_runner = ModelRunner(tokenizer=Tokenizer(model_path=os.path.join(model_path, "tokenizer.model")),
                               weights=load_weights(os.path.join(model_path, "consolidated.00.pth")),
                               params=load_model_params(os.path.join(model_path, "params.json")))

    # Load dataset
    dataset = load_dataset("openai/gsm8k", "main")['test']
    benchmark = MCTSBenchmark(dataset, model_runner, start_idx)
    benchmark.prepare_dataset(n_samples=10)
    prefix = json.load(open('prompts.json'))['repeated']['prompt']
    results = benchmark.run(prefix=prefix, rollouts=1, depth_limit=6, confidence=1, action_generation=1, use_aggregate=use_aggregate)

    # Save results
    output_file = f'gsm8k_benchmark.json'
    with open(output_file, 'w') as f:
        json.dump(vars(results), f, indent=2)

    print(f"\nFinal accuracy: {results.accuracy:.2%}")
    print(f"Last processed index: {results.last_index}")

if __name__ == "__main__":
    import sys
    main()
