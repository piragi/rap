import json
import os
from typing import NamedTuple, Optional

class ModelParams(NamedTuple):
    dim: int
    n_layers: int
    n_local_heads: int
    n_local_kv_heads: int
    head_dim: int
    vocab_size: int
    multiple_of: int
    ffn_dim_multiplier: Optional[float]
    norm_eps: float
    rope_theta: float
    use_scaled_rope: bool
    max_seq_len: int

def load_model_params(params_path: str) -> ModelParams:
    with open(params_path, "r") as f:
        params_dict = json.load(f)

    return ModelParams(
        dim=params_dict['dim'],
        n_layers=params_dict['n_layers'],
        n_local_heads=params_dict['n_heads'],
        n_local_kv_heads=params_dict.get('n_kv_heads'),
        head_dim=params_dict['dim'] // params_dict['n_heads'],
        vocab_size=params_dict['vocab_size'],
        multiple_of=params_dict['multiple_of'],
        ffn_dim_multiplier=params_dict.get('ffn_dim_multiplier'),  # Optional field
        norm_eps=params_dict['norm_eps'],
        rope_theta=params_dict['rope_theta'],
        use_scaled_rope=params_dict['use_scaled_rope'],
        max_seq_len=4096)
