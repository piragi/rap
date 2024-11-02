from typing import List, NamedTuple

import torch

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

class LayerWeights(NamedTuple):
    wq: torch.Tensor
    wk: torch.Tensor
    wv: torch.Tensor
    wo: torch.Tensor
    w1: torch.Tensor
    w2: torch.Tensor
    w3: torch.Tensor
    ffn_norm: torch.Tensor
    attention_norm: torch.Tensor

class TransformerWeights(NamedTuple):
    tok_embeddings: torch.Tensor
    norm: torch.Tensor
    output: torch.Tensor
    layer_weights: List[LayerWeights]

def load_weights(checkpoint_path: str) -> TransformerWeights:
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    layer_weights = []
    for i in range(len([k for k in state_dict.keys() if k.endswith(".wq.weight")])):
        layer = LayerWeights(
            wq=state_dict[f"layers.{i}.attention.wq.weight"].to(device),
            wk=state_dict[f"layers.{i}.attention.wk.weight"].to(device),
            wv=state_dict[f"layers.{i}.attention.wv.weight"].to(device),
            wo=state_dict[f"layers.{i}.attention.wo.weight"].to(device),
            w1=state_dict[f"layers.{i}.feed_forward.w1.weight"].to(device),
            w2=state_dict[f"layers.{i}.feed_forward.w2.weight"].to(device),
            w3=state_dict[f"layers.{i}.feed_forward.w3.weight"].to(device),
            ffn_norm=state_dict[f"layers.{i}.ffn_norm.weight"].to(device),
            attention_norm=state_dict[f"layers.{i}.attention_norm.weight"].to(device),
        )
        layer_weights.append(layer)
    return TransformerWeights(
        tok_embeddings=state_dict["tok_embeddings.weight"].to(device),
        norm=state_dict["norm.weight"].to(device),
        output=state_dict["output.weight"].to(device),
        layer_weights=layer_weights,
    )
