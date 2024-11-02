import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import ModelParams
from weights import LayerWeights, TransformerWeights

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

class KVCache(nn.Module):
    def __init__(
        self,
        layers: int,
        bsz: int,
        max_seq_len: int,
        n_local_kv_heads: int,
        head_dim: int,
    ):
        super(KVCache, self).__init__()
        self.register_buffer(
            "k",
            torch.zeros(
                layers,
                bsz,
                max_seq_len,
                n_local_kv_heads,
                head_dim,
                dtype=torch.bfloat16,
                device=device,
            ),
        )
        self.register_buffer(
            "v",
            torch.zeros(
                layers,
                bsz,
                max_seq_len,
                n_local_kv_heads,
                head_dim,
                dtype=torch.bfloat16,
                device=device,
            ),
        )

    def update(
        self,
        xk: torch.Tensor,
        xv: torch.Tensor,
        layer_idx: int,
        curr_pos: int,
        n_rep: int,
    ):
        xk = xk.to(self.k.dtype)
        xv = xv.to(self.v.dtype)

        seq_len = xk.size(1)
        self.k[layer_idx, :, curr_pos:curr_pos + seq_len, :, :] = xk
        self.v[layer_idx, :, curr_pos:curr_pos + seq_len, :, :] = xv

        # repeat kv for grouped query attention
        keys = self.k[layer_idx, :, :curr_pos + seq_len, :, :]
        keys = keys.repeat_interleave(n_rep, dim=2)
        values = self.v[layer_idx, :, :curr_pos + seq_len, :, :]
        values = values.repeat_interleave(n_rep, dim=2)
        return keys, values, self

    def clear(self):
        self.k.zero_()
        self.v.zero_()

def rms_norm(x: torch.Tensor, w: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return w * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps))

def apply_rotary_emb(xq: torch.Tensor,
                     xk: torch.Tensor,
                     freq_cis: torch.Tensor,
                     dtype: torch.dtype = torch.float32) -> Tuple[torch.Tensor, torch.Tensor]:
    # reshape into real, img and view as complex
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 2)
    xq_ = torch.view_as_complex(xq_)
    xk_ = torch.view_as_complex(xk_)
    # combine with freq matrix and flatten
    xq_out = torch.view_as_real(xq_ * freq_cis.unsqueeze(0).unsqueeze(2)).flatten(3)
    xk_out = torch.view_as_real(xk_ * freq_cis.unsqueeze(0).unsqueeze(2)).flatten(3)
    return xq_out.to(dtype), xk_out.to(dtype)

def feed_forward(x: torch.Tensor, weights: LayerWeights) -> torch.Tensor:
    w1 = F.silu(F.linear(x, weights.w1))
    w3 = F.linear(x, weights.w3)
    w2 = F.linear(w1 * w3, weights.w2)
    return w2

def attention(
    x: torch.Tensor,
    weights: LayerWeights,
    kv_cache: KVCache,
    layer_idx: int,
    curr_pos: int,
    attention_mask: Optional[torch.Tensor],
    freq_cis: torch.Tensor,
    model_params: ModelParams,
) -> Tuple[torch.Tensor, KVCache, torch.Tensor]:
    bsz, seqlen, _ = x.shape
    n_rep = model_params.n_local_heads // model_params.n_local_kv_heads
    xq = F.linear(x, weights.wq).view(bsz, -1, model_params.n_local_heads, model_params.head_dim)
    xk = F.linear(x, weights.wk).view(bsz, -1, model_params.n_local_kv_heads, model_params.head_dim)
    xv = F.linear(x, weights.wv).view(bsz, -1, model_params.n_local_kv_heads, model_params.head_dim)
    xq, xk = apply_rotary_emb(xq, xk, freq_cis, dtype=torch.bfloat16)
    keys, values, kv_cache = kv_cache.update(xk, xv, layer_idx, curr_pos, n_rep)
    xq = torch.permute(xq, (0, 2, 1, 3))  # (bsz, n_heads, seq_len, head_dim)
    keys = torch.permute(keys, (0, 2, 3, 1))  # (bsz, n_heads, head_dim, cache_len)
    values = torch.permute(values, (0, 2, 1, 3))  # (bsz, n_heads, cache_len, head_dim)
    scores = torch.matmul(xq, keys)
    pre_scores = scores / math.sqrt(model_params.head_dim)
    scores = pre_scores.to(torch.float32)  # softmax only over torch.float32
    if curr_pos == 0:
        scores = scores + attention_mask
    scores = F.softmax(scores.float(), dim=-1).type_as(xq)
    output = torch.matmul(scores, values)
    output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
    out = F.linear(output, weights.wo)
    return out, kv_cache, scores

def transformer(
    weights: TransformerWeights,
    model_params: ModelParams,
    tokens: torch.Tensor,
    curr_pos: int,
    kv_cache: KVCache,
    freq_cis: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, KVCache, torch.Tensor]:
    h = weights.tok_embeddings[tokens]
    scores = torch.empty(0)

    for layer in range(model_params.n_layers):
        norm_x = rms_norm(h, weights.layer_weights[layer].attention_norm)
        h_attn, kv_cache, scores = attention(
            norm_x,
            weights.layer_weights[layer],
            kv_cache,
            layer,
            curr_pos,
            attention_mask,
            freq_cis,
            model_params,
        )
        h = h + h_attn
        h_rms_norm = rms_norm(h, weights.layer_weights[layer].ffn_norm)
        h = h + feed_forward(h_rms_norm, weights.layer_weights[layer])
    logits = F.linear(rms_norm(h, weights.norm), weights.output)
    return logits, kv_cache, scores
