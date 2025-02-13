import json
from collections import Counter
from typing import List, Optional, Tuple, Union

import torch

from config import ModelParams
from model import KVCache, transformer
from token_tracker import TokenUsageStats
from tokenizer import Tokenizer
from weights import TransformerWeights

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

def get_confidence_state(action: str,
                         max_samples: int,
                         tokenizer: Tokenizer,
                         transformer_weights: TransformerWeights,
                         model_params: ModelParams,
                         batch_size: int = 5,
                         token_stats: Optional[TokenUsageStats] = None) -> Tuple[torch.Tensor, float]:
    """Get model's confidence in generated answers through adaptive sampling.
    
    Samples multiple completions and returns the most consistent answer along
    with a confidence score based on agreement between samples.
    
    Args:
        action: The prompt to generate completions for
        max_samples: Maximum number of samples to generate
        tokenizer: Tokenizer for encoding/decoding
        transformer_weights: Model weights
        model_params: Model parameters
        batch_size: Number of parallel samples per batch
        token_stats: Optional token usage tracker
        
    Returns:
        Tuple of (tokens for most common answer, confidence score)
    """
    answers = []
    tokens_map = {}
    sampled = 0

    while sampled < max_samples:
        curr_batch_size = min(batch_size, max_samples - sampled)
        prompts = [action] * curr_batch_size
        batched_tokens = prepare_tokens(prompts, tokenizer)
        input_length = batched_tokens.size(1)
        gen_tokens = generate(transformer_weights,
                              model_params,
                              batched_tokens,
                              tokenizer,
                              max_gen_len=input_length + 200,
                              track_method="generate_state",
                              token_stats=token_stats)

        for tokens in gen_tokens:
            text = tokenizer.decode(tokens.tolist())
            parts = text.split("The answer is")
            if len(parts) > 1:
                answer_part = parts[-1]
                for terminator in ['.', '\n', '!', '?']:
                    if terminator in answer_part:
                        answer = answer_part.split(terminator)[0].strip()
                        break
                else:
                    answer = answer_part.strip()
                if answer:
                    answers.append(answer)
                    tokens_map[answer] = tokens

        sampled += curr_batch_size
        if not answers:
            continue

        counter = Counter(answers)
        most_common = counter.most_common(2)  # Get top 2 answers

        if len(most_common) > 0:
            top_count = most_common[0][1]
            if (top_count >= 2 and (len(most_common) == 1 or top_count > most_common[1][1]) and top_count >= len(answers) / 2):
                break

    if not answers:
        print(text)
        return None, 0.0
    most_common_answer, count = counter.most_common(1)[0]
    confidence = count / len(answers)
    return tokens_map[most_common_answer].unsqueeze(0), confidence

def get_self_eval(reasoning: Union[str, List[str]],
                  tokenizer: Tokenizer,
                  transformer_weights: TransformerWeights,
                  model_params: ModelParams,
                  token_stats: Optional[TokenUsageStats] = None) -> List[float]:
    yes_probs = []
    useful = json.load(open('rap/prompts.json'))['useful_noprompt']['prompt']
    for r in reasoning:
        prompt = f"{useful}\n{r}\nIs the new question useful? "
        tokens = torch.tensor([tokenizer.encode(prompt, bos=False, eos=False, allowed_special="all")], dtype=torch.long, device=device)

        # Track +1 for the single token generation (Yes/No)
        if token_stats:
            token_stats.add_generate_action(1)  # One token for Yes/No prediction

        seqlen = tokens.shape[1]
        attn_mask = build_attn_mask(seqlen, 0)
        freqs_cis = precompute_freqs_cis(
            model_params.head_dim,
            model_params.max_seq_len,
            model_params.rope_theta,
            model_params.use_scaled_rope,
        )
        kvcache = KVCache(
            model_params.n_layers,
            1,
            model_params.max_seq_len,
            model_params.n_local_kv_heads,
            model_params.head_dim,
        ).to(device)

        logits, _, _ = transformer(
            transformer_weights,
            model_params,
            tokens,
            0,
            kvcache,
            freqs_cis[:seqlen],
            attention_mask=attn_mask,
        )

        yes_token = tokenizer.encode(" Yes", bos=False, eos=False, allowed_special="all")[0]
        no_token = tokenizer.encode(" No", bos=False, eos=False, allowed_special="all")[0]

        batch_logits = logits[:, -1][:, [yes_token, no_token]]
        probs = torch.softmax(batch_logits, dim=-1)
        yes_probs.append(probs[0, 0].item())

    return yes_probs

def prepare_tokens(input_data, tokenizer: Tokenizer):
    """Helper to prepare tokens for generate function"""
    if isinstance(input_data, str):
        tokens = tokenizer.encode(input_data, bos=False, eos=False, allowed_special="all")
        return torch.tensor([tokens], device=device)
    elif isinstance(input_data, list):
        # Batch of strings
        tokens = [tokenizer.encode(x, bos=False, eos=False, allowed_special="all") for x in input_data]
        max_len = max(len(t) for t in tokens)
        padded = [t + [tokenizer.eos_id] * (max_len - len(t)) for t in tokens]
        return torch.tensor(padded, device=device)
    else:
        return input_data.to(device)  # Assume it's already a tensor

# from entropix repo
def generate(
    transformer_weights: TransformerWeights,
    model_params: ModelParams,
    tokens: torch.Tensor,
    tokenizer: Tokenizer,
    temperature: float = 0.8,
    top_p: float = 0.90,
    max_gen_len: int = 200,
    track_method: Optional[str] = None,
    token_stats: Optional[TokenUsageStats] = None,
) -> torch.Tensor:
    """Generate text using autoregressive sampling.
    
    Core generation function that handles batched token generation with caching,
    temperature sampling, and optional token tracking.
    
    Args:
        transformer_weights: Model weights
        model_params: Model configuration parameters
        tokens: Input token tensor of shape (batch_size, seq_len)
        tokenizer: Tokenizer for encoding/decoding
        temperature: Sampling temperature (higher = more random)
        top_p: Nucleus sampling threshold
        max_gen_len: Maximum number of tokens to generate
        track_method: Optional method name for token tracking
        token_stats: Optional token usage statistics tracker
        
    Returns:
        Tensor of generated tokens of shape (batch_size, seq_len + generated_len)
    """
    tokens = tokens.to(device)
    bsz, seqlen = tokens.shape
    curr_pos = 0

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

    attn_mask = build_attn_mask(seqlen, curr_pos)
    logits, kvcache, _ = transformer(
        transformer_weights,
        model_params,
        tokens,
        curr_pos,
        kvcache,
        freqs_cis[:seqlen],
        attention_mask=attn_mask,
    )

    probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
    next_token = sample_top_p(probs, top_p)
    generated = [next_token]

    finished = torch.zeros(bsz, dtype=torch.bool, device=device)
    curr_pos = seqlen

    stop = torch.Tensor(tokenizer.stop_tokens).to(device)

    while curr_pos < seqlen + max_gen_len and not finished.all():
        curr_pos += 1
        active = ~finished
        if active.any():
            logits, kvcache, _ = transformer(
                transformer_weights,
                model_params,
                next_token,
                curr_pos,
                kvcache,
                freqs_cis[curr_pos:curr_pos + 1],
            )
            probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
            next_candidates = sample_top_p(probs, top_p)
            next_token = torch.where(finished.unsqueeze(1), next_token, next_candidates)

            for i in range(bsz):
                if not finished[i]:
                    token_str = tokenizer.decode([next_token[i].item()])
                    if torch.isin(next_token, stop).any() or '\n' in token_str:
                        finished[i] = True
            generated.append(next_token)

    generated = torch.cat(generated, dim=1)

    if track_method and token_stats:
        # Count non-padding tokens in output
        for seq in generated:
            token_count = int((seq != tokenizer.pad_id).sum().item())
            if track_method == 'generate_action':
                token_stats.add_generate_action(token_count)
            elif track_method == 'generate_state':
                token_stats.add_generate_state(token_count)
            elif track_method == 'cot':
                token_stats.add_cot(token_count)
    return generated
