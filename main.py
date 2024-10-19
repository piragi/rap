import os

import torch

from config import ModelParams, load_model_params
from model import KVCache, transformer
from tokenizer import Tokenizer
from weights import load_weights

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

def apply_scaling(freqs: torch.Tensor) -> torch.Tensor:
    SCALE_FACTOR = 8.0
    LOW_FREQ_FACTOR = 1.0
    HIGH_FREQ_FACTOR = 4.0
    OLD_CONTEXT_LEN = 8192  # original llama3 length

    low_freq_wavelen = OLD_CONTEXT_LEN / LOW_FREQ_FACTOR
    high_freq_wavelen = OLD_CONTEXT_LEN / HIGH_FREQ_FACTOR

    def scale_freq(freq: torch.Tensor) -> torch.Tensor:
        wavelen = 2 * torch.pi / freq

        # Calculate smooth factor
        smooth = (OLD_CONTEXT_LEN / wavelen - LOW_FREQ_FACTOR) / (HIGH_FREQ_FACTOR - LOW_FREQ_FACTOR)
        smooth = torch.clamp(smooth, 0.0, 1.0)  # Ensure smooth is between 0 and 1

        # Calculate scaled frequency
        scaled = (1 - smooth) * freq / SCALE_FACTOR + smooth * freq

        # Apply conditional scaling
        scaled = torch.where(
            wavelen < high_freq_wavelen,
            freq,  # No scaling
            torch.where(
                wavelen > low_freq_wavelen,
                freq / SCALE_FACTOR,  # Apply scaling factor
                scaled,  # Apply smooth scaling
            ),
        )
        return scaled

    scaled_freqs = torch.vmap(scale_freq)(freqs)

    return scaled_freqs

def precompute_freqs_cis(
    dim: int,
    end: int,
    theta: float = 500000.0,
    use_scaled: bool = False,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    freqs = 1.0 / (theta**(torch.arange(0, dim, 2, dtype=dtype, device=device)[:(dim // 2)] / dim))
    if use_scaled:
        freqs = apply_scaling(freqs)

    t = torch.arange(end, dtype=dtype, device=device).unsqueeze(1)  # Shape: (end, 1)
    freqs = freqs.unsqueeze(0)  # Shape: (1, dim//2)
    freqs = t * freqs  # Broadcasting to shape: (end, dim//2)
    return torch.exp(1j * freqs)

def build_attn_mask(seqlen: int, start_pos: int) -> torch.Tensor:
    mask = None
    if seqlen > 1:
        mask = torch.full((seqlen, seqlen), float("-inf"))
        mask = torch.triu(mask, diagonal=1)
        mask = (torch.hstack([torch.zeros((seqlen, start_pos)), mask]).to(torch.float32).to(device))
    return mask

# Llama implementation
def sample_top_p(probs, p):
    """
    Perform top-p (nucleus) sampling on a probability distribution.

    Args:
        probs (torch.Tensor): Probability distribution tensor.
        p (float): Probability threshold for top-p sampling.

    Returns:
        torch.Tensor: Sampled token indices.

    Note:
        Top-p sampling selects the smallest set of tokens whose cumulative probability mass
        exceeds the threshold p. The distribution is renormalized based on the selected tokens.
    """
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token

home_dir = os.path.expanduser("~")
model_path = os.path.join(home_dir, ".llama", "checkpoints", "Llama3.2-3B-Instruct")
model_params = load_model_params(os.path.join(model_path, "params.json"))
transformer_weights = load_weights(os.path.join(model_path, "consolidated.00.pth"))
tokenizer = Tokenizer(model_path=os.path.join(model_path, "tokenizer.model"))
prompt4 = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a masterful story teller. you can paint with all the colors of the wind.<|eot_id|><|start_header_id|>user<|end_header_id|>

Tell me a long and wonderful story aboout the adventures of the elven mage frieren and her band of heros<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""

raw_tokens1 = tokenizer.encode(prompt4, bos=False, eos=False, allowed_special="all")

def generate(transformer_weights, model_params: ModelParams, tokens):
    gen_tokens = None
    cur_pos = 0
    tokens = torch.tensor([tokens], dtype=torch.long).to(device)
    bsz, seqlen = tokens.shape
    attn_mask = build_attn_mask(seqlen, cur_pos)
    freqs_cis = precompute_freqs_cis(
        model_params.head_dim,
        model_params.max_seq_len,
        model_params.rope_theta,
        model_params.use_scaled_rope,
    )
    kvcache = KVCache(
        model_params.n_layers,
        bsz,
        model_params.max_seq_len,
        model_params.n_local_kv_heads,
        model_params.head_dim,
    ).to(device)
    (
        logits,
        kvcache,
        _,
    ) = transformer(
        transformer_weights,
        model_params,
        tokens,
        cur_pos,
        kvcache,
        freqs_cis[:seqlen],
        attention_mask=attn_mask,
    )
    next_token = torch.argmax(logits[:, -1], dim=-1, keepdim=True).to(torch.int32)
    gen_tokens = next_token
    print(tokenizer.decode([next_token.item()]), end="", flush=True)
    cur_pos = seqlen
    stop = torch.tensor([128001, 128008, 128009], device=device, dtype=torch.int32)
    while cur_pos < 8192:
        cur_pos += 1
        logits, kvcache, scores = transformer(
            transformer_weights,
            model_params,
            next_token,
            cur_pos,
            kvcache,
            freqs_cis[cur_pos:cur_pos + 1],
        )
        probs = torch.softmax(logits[:, -1] / 0.666, dim=-1)
        next_token = sample_top_p(probs, 0.9)
        gen_tokens = torch.cat((gen_tokens, next_token), dim=1)
        print(tokenizer.decode(next_token.tolist()[0]), end="", flush=True)
        if torch.isin(next_token, stop).any():
            break

print(prompt4)
generate(transformer_weights, model_params, raw_tokens1)
