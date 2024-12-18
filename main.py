import os
from collections import Counter
from typing import List, Optional, Tuple, Union

import torch

from config import ModelParams, load_model_params
from model import KVCache, transformer
from tokenizer import Tokenizer
from weights import TransformerWeights, load_weights

MAX_SEQ_LEN = 8192
USEFUL_PREFIX = """Given a question and some sub-questions, determine whether the last sub-question is useful to answer the question. Output 'Yes' or 'No', and a reason.\n\nQuestion 1: Four years ago, Kody was only half as old as Mohamed. If Mohamed is currently twice as 30 years old, how old is Kody?\nQuestion 1.1: How old is Mohamed?\nQuestion 1.2: How old was Mohamed four years ago?\nNew question 1.3: How old was Kody four years ago?\nIs the new question useful? Yes. We need the answer to calculate how old is Kody now.\n\nQuestion 2: Traci and Harris are baking cakes together. Traci has brought flour from her own house and Harris has 400g of flour in his house. Each cake needs 100g of flour and Traci and Harris have created 9 cakes each. How much flour, in grams, did Traci bring from her own house?\nNew question 2.1: How many cakes did Traci bring from her own house?\nIs the new question useful? No. The new question is not related to the original question.\n\nQuestion 3: A quantity surveyor is figuring the construction costs for a couple that wishes to build a house. The costs are as follows: land costs $50 per square meter, bricks cost $100 per 1000 bricks and roof tiles cost $10 per roof tile. If the house they wish to build requires 2000 square meters, 10000 bricks, and 500 roof tiles, how much construction costs are required for this project?\nQuestion 3.1: How much does the land cost?\nQuestion 3.2: How much do the bricks cost?\nNew question 3.3: How much do the roof tiles cost?\nIs the new question useful? Yes. We need the answer to calculate the total construction costs.\n\nQuestion 4: Wallace's water heater is twice the size of Catherine's water heater. If the capacity of Wallace's water heater is 40 gallons and it's 3/4 full, calculate the total number of gallons of water they both have if Catherine's water heater is also full with water to 3/4 of its capacity.\nQuestion 4.1: How much water is in Wallace's water heater?\nNew question 4.2: How much water do they have in total?\nIs the new question useful? No. It is too hard to answer the new question based on the current information."""

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
                         model_params: ModelParams) -> Tuple[torch.Tensor, float]:
    """
    Get confidence state by running multiple generations and finding the most common answer.
    """

    prompts = [action] * confidence_iteration
    batched_tokens = prepare_tokens(prompts, tokenizer)
    input_length = batched_tokens.size(1)
    gen_tokens = generate(transformer_weights, model_params, batched_tokens, tokenizer, max_gen_len=input_length + 200)

    # Process all generated sequences
    answers = []
    for i in range(confidence_iteration):
        text = tokenizer.decode(gen_tokens[i].tolist())
        parts = text.split("The answer is")
        if len(parts) > 1:
            answer_part = parts[-1]
            # Look for the first sentence terminator
            for terminator in ['.', '\n', '!', '?']:
                if terminator in answer_part:
                    answer = answer_part.split(terminator)[0].strip()
                    break
            else:
                answer = answer_part.strip()
        else:
            answer = ""
        answers.append(answer)

    # Find most common answer
    counter = Counter(answers)
    most_common = counter.most_common(1)[0]
    most_common_answer, most_common_count = most_common
    confidence = most_common_count / confidence_iteration

    # Return the first generated sequence that matches the most common answer
    return gen_tokens[answers.index(most_common_answer)].unsqueeze(0), confidence

def get_self_eval(reasoning: Union[str, List[str]], new_subquestion: Union[str, List[str]], tokenizer: Tokenizer,
                  transformer_weights: TransformerWeights, model_params: ModelParams) -> List[float]:
    """
    Evaluate usefulness of subquestions in batch.
    
    Args:
        reasoning: Single reasoning string or list of reasoning strings
        new_subquestion: Single question or list of questions to evaluate
        
    Returns:
        If inputs are strings: returns single confidence float
        If inputs are lists: returns list of confidence floats
    """
    prompts = [f"{USEFUL_PREFIX}{r}{q}\nIs the new question useful? " for r, q in zip(reasoning, new_subquestion)]

    # Encode all prompts
    tokens_list = [tokenizer.encode(p, bos=False, eos=False, allowed_special="all") for p in prompts]
    # Pad to max length
    max_len = max(len(t) for t in tokens_list)
    padded_tokens = [t + [tokenizer.pad_id] * (max_len - len(t)) for t in tokens_list]
    tokens = torch.tensor(padded_tokens, dtype=torch.long, device=device)

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

    logits, kvcache, _ = transformer(
        transformer_weights,
        model_params,
        tokens,
        0,
        kvcache,
        freqs_cis[:seqlen],
        attention_mask=attn_mask,
    )

    yes_token = tokenizer.encode("Yes", bos=False, eos=False, allowed_special="all")[0]
    no_token = tokenizer.encode("No", bos=False, eos=False, allowed_special="all")[0]

    # Get probabilities for Yes/No for all sequences
    batch_logits = logits[:, -1][:, [yes_token, no_token]]  # Shape: [batch_size, 2]
    probs = torch.softmax(batch_logits, dim=-1)
    yes_probs = probs[:, 0].tolist()  # Yes probability for each sequence
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
) -> torch.Tensor:
    """
    Generate tokens from an input tensor.
    Assumes tokens is already properly batched and padded if needed.
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
    return torch.cat(generated, dim=1)

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
