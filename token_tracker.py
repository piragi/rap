import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List

@dataclass
class TokenUsageStats:
    """Track token usage statistics for different methods with buffering"""
    # Final stored tokens for successful runs
    generate_action_tokens: List[int] = field(default_factory=list)
    generate_state_tokens: List[int] = field(default_factory=list)
    cot_tokens: List[int] = field(default_factory=list)

    # Track which tokens belong to which run
    current_run: int = 0
    runs: Dict[int, Dict[str, List[int]]] = field(default_factory=lambda: defaultdict(lambda: defaultdict(list)))

    # Buffer for current run
    buffer: Dict[str, List[int]] = field(default_factory=lambda: defaultdict(list))
    cot_buffer: List[int] = field(default_factory=list)

    def start_new_run(self):
        """Start tracking a new MCTS run"""
        # Clear the buffer for the new run
        self.buffer.clear()
        self.cot_buffer.clear()
        self.current_run += 1

    def add_generate_action(self, count: int):
        """Add to buffer instead of directly to stats"""
        self.buffer['generate_action'].append(count)

    def add_generate_state(self, count: int):
        """Add to buffer instead of directly to stats"""
        self.buffer['generate_state'].append(count)

    def add_cot(self, count: int):
        """Add to CoT buffer instead of directly to stats"""
        self.cot_buffer.append(count)

    def commit_run(self):
        """Commit the buffered tokens to the actual stats"""
        if self.buffer:
            # Add buffered tokens to the permanent storage
            self.generate_action_tokens.extend(self.buffer.get('generate_action', []))
            self.generate_state_tokens.extend(self.buffer.get('generate_state', []))

            # Add to runs tracking
            self.runs[self.current_run]['generate_action'].extend(self.buffer.get('generate_action', []))
            self.runs[self.current_run]['generate_state'].extend(self.buffer.get('generate_state', []))

            # Clear buffer
            self.buffer.clear()

        if self.cot_buffer:
            # Sum up all CoT attempts for this run into a single value
            total_cot_tokens = sum(self.cot_buffer)
            self.cot_tokens.append(total_cot_tokens)
            # Add to runs tracking
            self.runs[self.current_run]['cot'] = [total_cot_tokens]
            # Clear CoT buffer
            self.cot_buffer.clear()

    def discard_run(self):
        """Discard the buffered tokens from a failed run"""
        self.buffer.clear()

    def get_stats(self) -> Dict:
        """Calculate statistics for all methods and total MCTS runs"""
        stats = {}

        # Individual method stats
        for method_name, tokens in [("generate_action", self.generate_action_tokens), ("generate_state", self.generate_state_tokens),
                                    ("cot", self.cot_tokens)]:
            if tokens:
                stats[method_name] = {
                    "mean": statistics.mean(tokens),
                    "median": statistics.median(tokens),
                    "total": sum(tokens),
                    "count": len(tokens)
                }

        # Calculate total MCTS stats per run
        if self.runs:
            total_tokens_per_run = []
            for run_id, run_data in self.runs.items():
                if 'cot' not in run_data:  # Only process MCTS runs
                    action_tokens = run_data.get('generate_action', [])
                    state_tokens = run_data.get('generate_state', [])

                    # Only count complete runs
                    if action_tokens and state_tokens:
                        run_total = sum(action_tokens) + sum(state_tokens)
                        total_tokens_per_run.append(run_total)

            if total_tokens_per_run:
                stats["mcts_total"] = {
                    "mean": statistics.mean(total_tokens_per_run),
                    "median": statistics.median(total_tokens_per_run),
                    "total": sum(total_tokens_per_run),
                    "count": len(total_tokens_per_run)
                }

        return stats
