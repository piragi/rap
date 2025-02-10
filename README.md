# Reproduction of Reasoning via Planning (RAP)

This repository contains a reproduction study of the RAP (Reasoning via Planning) framework, focusing on its practical applicability in open-ended mathematical reasoning. The study examines RAP's performance on the GSM8K benchmark and provides detailed analysis of its computational requirements compared to Chain-of-Thought baselines.

## Key Findings

- Successfully reproduced RAP's enhanced reasoning capabilities over Chain-of-Thought baselines
- Identified significant computational overhead: RAP requires 12-20x more tokens per iteration

## Requirements

- Python 3.8+
- Access to Meta's LLaMA model (requires URL from Meta)
- 16GB+ RAM recommended
- CUDA-capable GPU recommended for faster inference

## Setup

1. First, ensure you have access to Meta's LLaMA model. You'll need the download URL from Meta.

2. Set the Meta URL as an environment variable:
```bash
export META_URL="your_meta_url_here"
```

3. Run the setup script:
```bash
chmod +x setup.sh
./setup.sh
```

The setup script will:
- Create a Python virtual environment
- Install all required dependencies
- Download and set up the LLaMA model
- Prepare the GSM8K dataset
- Create an activation script for the environment

4. After setup, activate the environment:
```bash
source ~/activate_env.sh
```

## Running Benchmarks

The project supports two types of benchmarks:

1. Chain of Thought (CoT):
```bash
python benchmark.py cot [start_index]
```

2. Reasoning via Planning (RAP):
```bash
python benchmark.py rap [start_index]
```

Optional parameters:
- `start_index`: Starting index in the GSM8K dataset (default: 0)

Example:
```bash
python benchmark.py rap 100  # Starts RAP benchmark from index 100
```

### Configuration

You can modify benchmark parameters in `benchmark.py`:

```python
@dataclass
class BenchmarkConfig:
    n_samples: int = 1000        # Number of samples to process
    max_iteration: int = 5       # Maximum iterations for CoT
    batched_iterations: int = 5  # Batch size for iterations
    rollouts: int = 1           # Number of MCTS rollouts
    depth_limit: int = 6        # Maximum MCTS tree depth
    confidence: int = 1         # MCTS exploration parameter
    action_generation: int = 1   # Number of candidate actions per step
    use_aggregate: bool = False  # Whether to use answer aggregation
```

## Results

The benchmark will generate several output files:

- `gsm8k_cot_iterations{n}_results_start{idx}_samples{n}.json`: Results for Chain of Thought
- `gsm8k_rap_results_start{idx}_samples{n}.json`: Results for RAP
- `cot_trace_{idx}.txt` or `rap_trace_{idx}.txt`: Detailed trace files

## Performance Analysis

Our implementation achieves the following accuracies on GSM8K:

| Method | 1 Iteration | 3 Iterations | 5 Iterations | 7 Iterations |
|--------|-------------|--------------|--------------|--------------|
| CoT    | 17.5%      | 25.2%        | 30.1%        | 31.1%       |
| RAP    | 23.5%      | 27.7%        | 28.9%        | 33.7%       |

Token consumption per problem:
- CoT: 79-560 tokens
- RAP: 1,583-7,046 tokens

## Acknowledgments

This work builds upon the original RAP paper by Hao et al. Special thanks to the authors for their pioneering work in this area.
