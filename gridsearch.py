import json
import os
from itertools import product
from typing import Dict, List, Optional
from dataclasses import dataclass

from tqdm import tqdm

from benchmark import MCTSBenchmark, ModelRunner, BenchmarkResult
from main import Tokenizer, load_model_params, load_weights
from datasets import load_dataset

@dataclass
class GridSearchResult:
    """Dataclass to store grid search results"""
    parameters: Dict
    benchmark_result: BenchmarkResult
    error: Optional[str] = None

class GridSearchRunner:
    """Class to handle grid search over benchmark parameters"""
    def __init__(self, 
                 benchmark: MCTSBenchmark,
                 output_dir: str = 'grid_search_results'):
        self.benchmark = benchmark
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def load_prompts(self, prompt_file: str = 'prompts.json', 
                    prompt_names: List[str] = ["repeated"]) -> Dict[str, str]:
        """Load prompts from JSON file."""
        try:
            with open(prompt_file, 'r') as f:
                prompt_configs = json.load(f)

            prompts = {
                name: prompt_configs[name]['prompt'] 
                for name in prompt_names 
                if name in prompt_configs
            }

            missing = set(prompt_names) - set(prompt_configs.keys())
            if missing:
                print(f"Warning: Prompts not found: {missing}")

            return prompts
        except Exception as e:
            print(f"Error loading prompts from {prompt_file}: {str(e)}")
            raise

    def run(self, param_grid: Dict[str, List], n_samples: int = 100) -> Dict[str, GridSearchResult]:
        """Run grid search over specified parameters."""
        # Load prompts and add to parameter grid
        prompts = self.load_prompts()
        param_grid['prompt'] = list(prompts.keys())

        # Generate parameter combinations
        param_names = sorted(param_grid.keys())
        param_values = [param_grid[name] for name in param_names]
        param_combinations = list(product(*param_values))

        results = {}

        for params in tqdm(param_combinations, desc="Parameter combinations"):
            param_dict = dict(zip(param_names, params))
            param_str = '_'.join(f"{k}-{v}" for k, v in param_dict.items())

            print(f"\nRunning combination: {param_str}")

            try:
                # Prepare benchmark parameters
                current_prompt = prompts[param_dict['prompt']]
                
                # Update dataset size
                self.benchmark.prepare_dataset(n_samples)
                
                # Run benchmark
                benchmark_result = self.benchmark.run(
                    prefix=current_prompt,
                    rollouts=param_dict['rollouts'],
                    depth_limit=param_dict['depth_limit'],
                    action_generation=param_dict['action_generation'],
                    confidence=param_dict['confidence'],
                    use_aggregate=True
                )

                # Store results
                results[param_str] = GridSearchResult(
                    parameters=param_dict,
                    benchmark_result=benchmark_result
                )

                # Save individual result
                self._save_result(param_str, param_dict, benchmark_result)
                
                print(f"Accuracy for {param_str}: {benchmark_result.accuracy:.2%}")

            except Exception as e:
                print(f"Error with parameters {param_str}: {str(e)}")
                results[param_str] = GridSearchResult(
                    parameters=param_dict,
                    benchmark_result=None,
                    error=str(e)
                )

        # Save combined results
        self._save_combined_results(results)
        return results

    def _save_result(self, param_str: str, params: Dict, result: BenchmarkResult) -> None:
        """Save individual result to file."""
        output_file = os.path.join(self.output_dir, f'results_{param_str}.json')
        with open(output_file, 'w') as f:
            json.dump({
                'parameters': params,
                'results': vars(result)
            }, f, indent=2)

    def _save_combined_results(self, results: Dict[str, GridSearchResult]) -> None:
        """Save combined results to file."""
        combined_results = {
            param_str: {
                'parameters': result.parameters,
                'accuracy': result.benchmark_result.accuracy if result.benchmark_result else None,
                'error': result.error
            }
            for param_str, result in results.items()
        }
        
        output_file = os.path.join(self.output_dir, 'combined_results.json')
        with open(output_file, 'w') as f:
            json.dump(combined_results, f, indent=2)

    def get_best_parameters(self, results: Dict[str, GridSearchResult]) -> Optional[Dict]:
        """Find the best performing parameters."""
        best_accuracy = 0
        best_params = None

        for result in results.values():
            if (result.benchmark_result and 
                result.benchmark_result.accuracy > best_accuracy):
                best_accuracy = result.benchmark_result.accuracy
                best_params = result.parameters

        return best_params

def main():
    # Load model components and create model runner
    home_dir = os.path.expanduser("~")
    model_path = os.path.join(home_dir, ".llama", "checkpoints", "Llama3.2-3B")
    
    model_runner = ModelRunner(
        tokenizer=Tokenizer(model_path=os.path.join(model_path, "tokenizer.model")),
        weights=load_weights(os.path.join(model_path, "consolidated.00.pth")),
        params=load_model_params(os.path.join(model_path, "params.json"))
    )

    # Load dataset and create benchmark
    dataset = load_dataset("openai/gsm8k", "main")['test']
    benchmark = MCTSBenchmark(dataset, model_runner)

    # Create grid search runner
    grid_search = GridSearchRunner(benchmark, output_dir='rap_grid_search_results')

    # Define parameter grid
    param_grid = {
        'rollouts': [3, 5, 7],
        'depth_limit': [6],
        'action_generation': [4],
        'confidence': [5]
    }

    # Run grid search
    results = grid_search.run(param_grid, n_samples=500)

    # Get and print best parameters
    best_params = grid_search.get_best_parameters(results)
    if best_params:
        print("\nBest performing parameters:")
        for param, value in best_params.items():
            print(f"{param}: {value}")
        best_result = next(r for r in results.values() 
                         if r.parameters == best_params)
        print(f"Accuracy: {best_result.benchmark_result.accuracy:.2%}")

if __name__ == "__main__":
    main()
