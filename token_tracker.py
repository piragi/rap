from dataclasses import dataclass, field
from typing import List, Dict
import statistics
from collections import defaultdict

@dataclass
class TokenUsageStats:
    """Track token usage statistics for different methods"""
    # Store tokens per run for each method
    generate_action_tokens: List[int] = field(default_factory=list)
    get_confidence_tokens: List[int] = field(default_factory=list)
    generate_state_tokens: List[int] = field(default_factory=list)
    cot_tokens: List[int] = field(default_factory=list)
    
    # Track which tokens belong to which run
    current_run: int = 0
    runs: Dict[int, Dict[str, List[int]]] = field(default_factory=lambda: defaultdict(lambda: defaultdict(list)))
    
    def start_new_run(self):
        """Start tracking a new MCTS run"""
        self.current_run += 1
    
    def add_generate_action(self, count: int):
        self.generate_action_tokens.append(count)
        self.runs[self.current_run]['generate_action'].append(count)
        
    def add_generate_state(self, count: int):
        self.generate_state_tokens.append(count)
        self.runs[self.current_run]['generate_state'].append(count)
        
    def add_cot(self, count: int):
        self.cot_tokens.append(count)
        self.runs[self.current_run]['cot'].append(count)
    
    def get_stats(self) -> Dict:
        """Calculate statistics for all methods and total MCTS runs"""
        stats = {}
        
        # Individual method stats
        for method_name, tokens in [
            ("generate_action", self.generate_action_tokens),
            ("generate_state", self.generate_state_tokens),
            ("cot", self.cot_tokens)
        ]:
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
                    run_total = (
                        sum(run_data.get('generate_action', [0])) +
                        sum(run_data.get('generate_state', [0]))
                    )
                    if run_total > 0:
                        total_tokens_per_run.append(run_total)
            
            if total_tokens_per_run:
                stats["mcts_total"] = {
                    "mean": statistics.mean(total_tokens_per_run),
                    "median": statistics.median(total_tokens_per_run),
                    "total": sum(total_tokens_per_run),
                    "count": len(total_tokens_per_run)
                }
            
        return stats
