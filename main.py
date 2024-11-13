import os
from collections import Counter
from typing import Optional

import torch

from config import ModelParams, load_model_params
from model import KVCache, transformer
from tokenizer import Tokenizer
from weights import TransformerWeights, load_weights

MAX_SEQ_LEN = 8192

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

# from entropix repo
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

# from entropix repo
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

def build_attn_mask(seqlen: int, start_pos: int) -> Optional[torch.Tensor]:
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

def get_action_loglikelihood(prefix: str, actions: list[str], tokenizer: Tokenizer, transformer_weights: TransformerWeights,
                             model_params: ModelParams):
    curr_pos = 0
    bsz = len(actions)
    assert bsz <= model_params.max_batch_size, (bsz, model_params.max_batch_size)

    prefix_tokens = tokenizer.encode(prefix, bos=False, eos=False, allowed_special="all")
    prompts_tokens = [tokenizer.encode(x, bos=False, eos=False, allowed_special="all") for x in actions]
    max_seq_len = max(len(prefix_tokens) + len(t) for t in prompts_tokens)

    sequences = [prefix_tokens + prompt + [tokenizer.pad_id] * (max_seq_len - len(prefix_tokens) - len(prompt)) for prompt in prompts_tokens]
    tokens = torch.tensor(sequences, device=device)

    attn_mask = build_attn_mask(max_seq_len, curr_pos)
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
        curr_pos,
        kvcache,
        freqs_cis[:max_seq_len],
        attention_mask=attn_mask,
    )

    acc_probs = torch.zeros(bsz, dtype=torch.float32).to(device)
    #TODO: im not so sure about adding probabilites?
    for i in range(len(prefix_tokens), max_seq_len):
        probs = torch.softmax(logits[:, i - 1, :], dim=-1)
        valid_tokens = tokens[:, i] != tokenizer.pad_id
        acc_probs += torch.log(probs[torch.arange(bsz), tokens[:, i]]) * valid_tokens
    return acc_probs

def get_confidence_state(action: str, confidence_iteration: int, tokenizer: Tokenizer, transformer_weights: TransformerWeights,
                         model_params: ModelParams):
    #TODO: we could do batched generation, need to change generate function first
    action_tokens = tokenizer.encode(action, bos=True, eos=False, allowed_special="all")
    states = []
    answers = []
    for _ in range(confidence_iteration):
        gen_tokens = generate(transformer_weights, model_params, action_tokens, tokenizer)
        text = tokenizer.decode(gen_tokens[0].tolist())
        answer = text.split("The answer is")[-1].split(".")[0].strip()
        answers.append(answer)
        states.append(gen_tokens)
    most_common = Counter(answers).most_common(1)[0][0]
    return states[answers.index(most_common)]

def get_self_eval(reasoning: str, tokenizer: Tokenizer, transformer_weights: TransformerWeights, model_params: ModelParams):
    prompt = f"{reasoning}\nIs this reasoning step correct?\n"
    prefix_tokens = tokenizer.encode(prompt, bos=True, eos=False, allowed_special="all")
    yes_token = tokenizer.encode("Yes", bos=False, eos=False, allowed_special="all")[0]

    tokens = torch.tensor([prefix_tokens], device=device)
    bsz, seqlen = tokens.shape
    attn_mask = build_attn_mask(seqlen, 0)
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
        0,
        kvcache,
        freqs_cis[:seqlen],
        attention_mask=attn_mask,
    )

    probs = torch.softmax(logits[0, -1], dim=-1)
    yes_prob = probs[yes_token].item()

    return yes_prob

# from entropix repo
def generate(transformer_weights: TransformerWeights,
             model_params: ModelParams,
             tokens: torch.Tensor,
             tokenizer: Tokenizer,
             temperature: float = 0.8,
             top_p: float = 0.9,
             print_generated: bool = False,
             gen_length: int = MAX_SEQ_LEN) -> torch.Tensor:
    gen_tokens = None
    curr_pos = 0
    tokens = torch.tensor([tokens], dtype=torch.long).to(device)
    bsz, seqlen = tokens.shape
    attn_mask = build_attn_mask(seqlen, curr_pos)
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
        curr_pos,
        kvcache,
        freqs_cis[:seqlen],
        attention_mask=attn_mask,
    )
    next_token = torch.argmax(logits[:, -1], dim=-1, keepdim=True).to(torch.int32)
    gen_tokens = next_token
    if print_generated: print(tokenizer.decode([next_token.item()]), end="", flush=True)
    curr_pos = seqlen
    stop = torch.Tensor(tokenizer.stop_tokens).to(device)
    while curr_pos < gen_length:
        curr_pos += 1
        logits, kvcache, _ = transformer(
            transformer_weights,
            model_params,
            next_token,
            curr_pos,
            kvcache,
            freqs_cis[curr_pos:curr_pos + 1],
        )

        probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
        next_token = sample_top_p(probs, top_p)
        gen_tokens = torch.cat((gen_tokens, next_token), dim=1)
        if print_generated: print(tokenizer.decode(next_token.tolist()[0]), end="", flush=True)

        if torch.isin(next_token, stop).any() or '\n' in tokenizer.decode(next_token.tolist()[0]): break
    return gen_tokens

if __name__ == "__main__":
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
    print(prompt4)
    generate(transformer_weights, model_params, raw_tokens1, tokenizer)
