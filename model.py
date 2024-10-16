from typing import Tuple

import torch
import torch.nn.functional as F

from config import ModelParams
from weights import LayerWeights, TransformerWeights

def rms_norm():
    raise NotImplementedError()

def rotary_embedding():
    raise NotImplementedError()

def feed_forward(x: torch.Tensor, weights) -> torch.Tensor:
    w1 = F.silu(F.linear(x, weights.x1))
    w3 = F.linear(x, weights.x3)
    w2 = F.linear(w1 * w3, weights.x2)
    return w2

def attention(x: torch.Tensor, weights: LayerWeights, model_params: ModelParams) -> Tuple[torch.Tensor, torch.Tensor]:
    bsz, seqlen, _ = x.shape
    xq = F.linear(x, weights.wq)
    xk = F.linear(x, weights.wk)
    xv = F.linear(x, weights.wv)
    raise NotImplementedError()

def transformer(weights, model_params, tokens):
    raise NotImplementedError()
