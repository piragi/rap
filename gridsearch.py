import json
import os
from itertools import product
from typing import Dict, List

import torch
from datasets import load_dataset
from tqdm import tqdm

from benchmark import run_benchmark
from config import ModelParams
from main import Tokenizer, load_model_params, load_weights

def load_prompts(prompt_file: str = 'prompts.json', prompt_names: List[str] = ["original", "repeated"]) -> Dict[str, str]:
    """
    Load prompts from JSON file.
    
    Args:
        prompt_file: Path to the JSON file containing prompts
        prompt_names: List of prompt names to extract. If None, extracts all prompts.
    
    Returns:
        Dictionary mapping prompt names to prompt texts
    """
    try:
        with open(prompt_file, 'r') as f:
            prompt_configs = json.load(f)

        prompts = {name: prompt_configs[name]['prompt'] for name in prompt_names if name in prompt_configs}

        # Warn if any requested prompts were not found
        missing_prompts = set(prompt_names) - set(prompt_configs.keys())
        if missing_prompts:
            print(f"Warning: The following prompts were not found: {missing_prompts}")

        return prompts
    except Exception as e:
        print(f"Error loading prompts from {prompt_file}: {str(e)}")
        raise

def run_grid_search(test_dataset,
                    tokenizer: Tokenizer,
                    transformer_weights: dict,
                    model_params: ModelParams,
                    param_grid: Dict[str, List],
                    n_samples: int = 100,
                    start_idx: int = 0,
                    output_dir: str = 'grid_search_results') -> Dict:
    """
    Run grid search over specified parameters.
    
    Args:
        test_dataset: The dataset to evaluate on
        tokenizer: Tokenizer instance
        transformer_weights: Model weights
        model_params: Model parameters
        param_grid: Dictionary of parameters and their values to search over
        n_samples: Number of samples to evaluate
        start_idx: Starting index in dataset
        output_dir: Directory to save results
    
    Returns:
        Dictionary containing results for all parameter combinations
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load prompts
    prompts = load_prompts()
    param_grid['prompt'] = list(prompts.keys())

    # Generate all combinations of parameters
    param_names = sorted(param_grid.keys())
    param_values = [param_grid[name] for name in param_names]
    param_combinations = list(product(*param_values))

    all_results = {}

    # Run benchmark for each parameter combination
    for params in tqdm(param_combinations, desc="Parameter combinations"):
        param_dict = dict(zip(param_names, params))

        # Create a descriptive identifier for this combination
        param_str = '_'.join(f"{k}-{v}" for k, v in param_dict.items())

        print(f"\nRunning combination: {param_str}")

        try:
            # Get the actual prompt text for this run
            current_prompt = prompts[param_dict['prompt']]

            # Run benchmark with current parameters
            results = run_benchmark(
                dataset=test_dataset,
                tokenizer=tokenizer,
                transformer_weights=transformer_weights,
                model_params=model_params,
                n_samples=n_samples,
                rollouts=param_dict['rollouts'],
                depth_limit=param_dict['depth_limit'],
                action_generation=param_dict['action_generation'],
                start_idx=start_idx,
                prefix=current_prompt  # Use the actual prompt text
            )

            # Save individual result
            output_file = os.path.join(output_dir, f'results_{param_str}.json')
            with open(output_file, 'w') as f:
                json.dump({'parameters': param_dict, 'results': results}, f, indent=2)

            # Store in combined results
            all_results[param_str] = {
                'parameters': param_dict,
                'accuracy': results['accuracy'],
                'correct': results['correct'],
                'total': results['total']
            }

            # Print current results
            print(f"Accuracy for {param_str}: {results['accuracy']:.2%}")

        except Exception as e:
            print(f"Error with parameters {param_str}: {str(e)}")
            all_results[param_str] = {'parameters': param_dict, 'error': str(e)}

    # Save combined results
    combined_output_file = os.path.join(output_dir, 'combined_results.json')
    with open(combined_output_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    return all_results

if __name__ == "__main__":
    # Load model components
    home_dir = os.path.expanduser("~")
    model_path = os.path.join(home_dir, ".llama", "checkpoints", "Llama3.2-3B-Instruct")
    model_params = load_model_params(os.path.join(model_path, "params.json"))
    transformer_weights = load_weights(os.path.join(model_path, "consolidated.00.pth"))
    tokenizer = Tokenizer(model_path=os.path.join(model_path, "tokenizer.model"))

    # Load GSM8K dataset
    dataset = load_dataset("openai/gsm8k", "main")
    test_dataset = dataset['test']

    # Define parameter grid
    param_grid = {'rollouts': [1], 'depth_limit': [6], 'action_generation': [1]}

    # Run grid search
    results = run_grid_search(
        test_dataset=test_dataset,
        tokenizer=tokenizer,
        transformer_weights=transformer_weights,
        model_params=model_params,
        param_grid=param_grid,
        n_samples=100,  # Adjust as needed
        start_idx=0,
        output_dir='rap_grid_search_results')

    # Print best results
    best_accuracy = 0
    best_params = None

    for param_str, result in results.items():
        if 'accuracy' in result and result['accuracy'] > best_accuracy:
            best_accuracy = result['accuracy']
            best_params = result['parameters']

    if best_params:
        print("\nBest performing parameters:")
        print(f"Parameters: {best_params}")
        print(f"Accuracy: {best_accuracy:.2%}")
