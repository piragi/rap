import json
import time
import os
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass
from collections import defaultdict

import torch
from datasets import load_dataset
from tqdm import tqdm

from config import ModelParams
from main import Tokenizer, generate, load_model_params, load_weights
from mcts import mcts
from weights import TransformerWeights
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
    def __init__(self, tokenizer: Tokenizer, weights: TransformerWeights, params: ModelParams):
        self.tokenizer = tokenizer
        self.weights = weights
        self.params = params

    def generate_response(self, prompt: str, temperature: float = 0.8) -> str:
        tokens = self.tokenizer.encode(prompt, bos=False, eos=False, allowed_special="all")
        tokens = torch.tensor([tokens]).cuda()
        output_tokens = generate(self.weights, self.params, tokens, 
                               self.tokenizer, temperature=temperature, 
                               max_gen_len=2048)
        return self.tokenizer.decode(output_tokens[0].tolist())

class BenchmarkRunner:
    """Base class for running benchmarks"""
    def __init__(self, dataset, model_runner: ModelRunner, start_idx: int = 0):
        self.dataset = dataset
        self.model_runner = model_runner
        self.start_idx = start_idx

    def prepare_dataset(self, n_samples: Optional[int]) -> None:
        if n_samples and n_samples > 0:
            end_idx = min(self.start_idx + n_samples, len(self.dataset))
            self.dataset = self.dataset.select(range(self.start_idx, end_idx))
        else:
            self.dataset = self.dataset.select(range(self.start_idx, len(self.dataset)))

class MCTSBenchmark(BenchmarkRunner):
    """MCTS-specific benchmark implementation"""
    def run(self, prefix: str, rollouts: int = 10, depth_limit: int = 5,
            confidence: int = 1, action_generation: int = 5, 
            trace_file: str = "mcts_trace.txt", use_aggregate: bool = False) -> BenchmarkResult:
        
        correct_mcts = correct_agg = total = 0
        results = []
        start_time = time.time()

        for idx, example in tqdm(enumerate(self.dataset, start=self.start_idx)):
            target = parse_target(example)
            if target is None:
                continue

            write_trace(trace_file, idx, example['question'], target)
            
            try:
                result = self._process_example(example, target, prefix, rollouts, 
                                            depth_limit, action_generation, confidence,
                                            use_aggregate)
                
                if result:
                    results.append(result)
                    correct_mcts += int(result.get('mcts_correct', False))
                    correct_agg += int(result.get('aggregate_correct', False))
                    total += 1

            except Exception as e:
                print(f"\nError processing example {idx}: {str(e)}")
                continue

        return BenchmarkResult(
            accuracy=correct_mcts / total if total > 0 else 0,
            correct=correct_mcts,
            total=total,
            results=results,
            last_index=idx,
            total_time_seconds=time.time() - start_time,
            aggregate_accuracy=correct_agg / total if use_aggregate and total > 0 else None,
            aggregate_correct=correct_agg if use_aggregate else None
        )

    def _process_example(self, example: Dict, target: float, prefix: str, 
                        rollouts: int, depth_limit: int, action_generation: int,
                        confidence: int, use_aggregate: bool) -> Optional[Dict]:
        init_state = State(states=[], prefix=prefix, question=example['question'])
        final_state, root = mcts(init_state, rollouts, depth_limit, action_generation,
                               self.model_runner.tokenizer, self.model_runner.weights,
                               self.model_runner.params, confidence)

        if not final_state or not final_state.states:
            return None

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
                'mcts_steps': [f"{state.subquestion}\n{state.subanswer}" 
                             for state in final_state.states]
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

class ChainOfThoughtBenchmark(BenchmarkRunner):
    """Chain of thought specific benchmark implementation"""
    def run(self, max_iterations: int = 10, 
            trace_file: str = "cot_trace.txt") -> BenchmarkResult:
        
        correct = total = 0
        results = []

        for idx, example in tqdm(enumerate(self.dataset, start=self.start_idx)):
            target = parse_target(example)
            if target is None:
                continue

            write_trace(trace_file, idx, example['question'], target)

            try:
                result = self._process_example(example, target, max_iterations)
                if result:
                    results.append(result)
                    correct += int(result['final_correct'])
                    total += 1

            except Exception as e:
                print(f"\nError processing example {idx}: {str(e)}")
                continue

        return BenchmarkResult(
            accuracy=correct / total if total > 0 else 0,
            correct=correct,
            total=total,
            results=results,
            last_index=idx
        )

    def _process_example(self, example: Dict, target: float, 
                        max_iterations: int) -> Optional[Dict]:
        predictions = []
        all_attempts = []
        
        for attempt in range(max_iterations):
            try:
                response = self._generate_attempt(example['question'], attempt)
                pred = extract_answer(response)
                
                if pred is not None:
                    predictions.append(pred)
                    all_attempts.append({
                        'attempt_number': attempt + 1,
                        'reasoning': response,
                        'predicted': pred,
                    })
                    
            except Exception as e:
                print(f"Error in attempt {attempt + 1}: {str(e)}")
                all_attempts.append({
                    'attempt_number': attempt + 1,
                    'error': str(e)
                })

        if not predictions:
            return None

        majority_pred = max(set(predictions), key=predictions.count)
        is_correct = abs(majority_pred - target) < 1e-6

        return {
            'index': example.get('id', -1),
            'question': example['question'],
            'target': target,
            'attempts': all_attempts,
            'predictions': predictions,
            'majority_prediction': majority_pred,
            'final_correct': is_correct
        }

    def _generate_attempt(self, question: str, attempt: int) -> str:
        prompt = self._get_prompt() + question + "\nA: "
        response = self.model_runner.generate_response(prompt)
        return response.split(prompt)[-1].strip()

    def _get_prompt(self) -> str:
        # Return the prompt template - moved to a separate method for clarity
        return """Q: [Example questions and answers...]\n\n"""  # Add your actual prompt template here

def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ['cot', 'rap']:
        print("Usage: python3 benchmark.py [cot|rap] [start_index] [--aggregate]")
        sys.exit(1)

    start_idx = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    use_aggregate = "--aggregate" in sys.argv

    # Initialize model
    home_dir = os.path.expanduser("~")
    model_path = os.path.join(home_dir, ".llama", "checkpoints", "Llama3.2-3B")
    model_runner = ModelRunner(
        tokenizer=Tokenizer(model_path=os.path.join(model_path, "tokenizer.model")),
        weights=load_weights(os.path.join(model_path, "consolidated.00.pth")),
        params=load_model_params(os.path.join(model_path, "params.json"))
    )

    # Load dataset
    dataset = load_dataset("openai/gsm8k", "main")['test']

    if sys.argv[1] == 'cot':
        benchmark = ChainOfThoughtBenchmark(dataset, model_runner, start_idx)
        benchmark.prepare_dataset(n_samples=500)
        results = benchmark.run(max_iterations=5)
        output_file = f'gsm8k_cot_results_start:{start_idx}_iterations:5.json'
    else:
        benchmark = MCTSBenchmark(dataset, model_runner, start_idx)
        benchmark.prepare_dataset(n_samples=100)
        prefix = json.load(open('prompts.json'))['repeated']['prompt']
        results = benchmark.run(prefix=prefix, rollouts=3, depth_limit=6,
                              confidence=5, action_generation=3,
                              use_aggregate=use_aggregate)
        output_file = f'gsm8k_rap_results_start:{start_idx}_rollouts:3.json'

    # Save results
    with open(output_file, 'w') as f:
        json.dump(vars(results), f, indent=2)

    print(f"\nFinal accuracy: {results.accuracy:.2%}")
    print(f"Last processed index: {results.last_index}")

if __name__ == "__main__":
    import sys
    main()
