from typing import NamedTuple, Tuple

import torch

from config import ModelParams
from main import generate, get_confidence_state, get_self_eval
from tokenizer import Tokenizer
from weights import TransformerWeights

Action = str

class SubStep(NamedTuple):
    subquestion: Action
    subanswer: str
    confidence: float

class State(NamedTuple):
    states: list[SubStep]
    prefix: str
    question: str

def advance_tokens(prompt: Union[str, List[str]], tokenizer: Tokenizer, transformer_weights: TransformerWeights, model_params: ModelParams):
    """
    Generate tokens for one or multiple prompts.
    
    Args:
        prompt: Single prompt string or list of prompts
        tokenizer: Tokenizer instance
        transformer_weights: Model weights
        model_params: Model parameters
        
    Returns:
        If input is string: returns generated text string
        If input is list: returns list of generated text strings
    """
    if isinstance(prompt, str):
        tokens = tokenizer.encode(prompt, bos=False, eos=False, allowed_special='all')
        is_single = True
    else:
        tokens = [tokenizer.encode(p, bos=False, eos=False, allowed_special='all') for p in prompt]
        # Pad to max length
        max_len = max(len(t) for t in tokens)
        tokens = [t + [tokenizer.pad_id] * (max_len - len(t)) for t in tokens]
        is_single = False

    generated_tokens = generate(transformer_weights, model_params, tokens, tokenizer)
    torch.cuda.empty_cache()

    if is_single:
        return tokenizer.decode(generated_tokens[0].tolist()).strip()
    else:
        return [tokenizer.decode(tokens.tolist()).strip() for tokens in generated_tokens]

def predict_action(state: State,
                   tokenizer: Tokenizer,
                   transformer_weights: TransformerWeights,
                   model_params: ModelParams,
                   batch_size: Optional[int] = None) -> Union[Tuple[Action, float], List[Tuple[Action, float]]]:
    """
    Predict next action(s) given current state.
    
    Args:
        state: Current state
        tokenizer: Tokenizer instance 
        transformer_weights: Model weights
        model_params: Model parameters
        batch_size: If provided, generate this many actions at once
        
    Returns:
        If batch_size is None: returns (action, confidence) tuple
        If batch_size is int: returns list of (action, confidence) tuples
    """
    s_t0 = state.prefix + f"\n\n{state.question}"
    s_t0 += ''.join(f'\nQuestion 5.{i+1}: {substate.subquestion}\nAnswer 5.{i+1}: {substate.subanswer}' for i, substate in enumerate(state.states))
    action = f'\nQuestion 5.{len(state.states)+1}: '

    if batch_size is None:
        # Single prediction
        subquestion = advance_tokens(s_t0 + action, tokenizer, transformer_weights, model_params)
        a_t0 = s_t0 + f'\nNew question 5.{len(state.states)+1}: {subquestion}'
        confidence = get_self_eval(a_t0, subquestion, tokenizer, transformer_weights, model_params)
        return subquestion, confidence
    else:
        # Batch prediction
        prompts = [s_t0 + action] * batch_size
        subquestions = advance_tokens(prompts, tokenizer, transformer_weights, model_params)

        # Batch evaluate all subquestions at once
        a_t0s = [s_t0 + f'\nNew question 5.{len(state.states)+1}: {subq}' for subq in subquestions]
        confidences = get_self_eval(a_t0s, subquestions, tokenizer, transformer_weights, model_params)

        return list(zip(subquestions, confidences))

def predict_state(state: State, action: Action, tokenizer: Tokenizer, transformer_weights: TransformerWeights, model_params: ModelParams) -> State:
    """Generate next state given an action"""
    s_t0 = state.prefix + f"\n\n{state.question}"
    s_t0 += ''.join(f'\nQuestion 5.{i+1}: {substate.subquestion}\nAnswer 5.{i+1}: {substate.subanswer}' for i, substate in enumerate(state.states))
    s_t0 += f'\nQuestion 5.{len(state.states)+1}: {action}\nAnswer 5.{len(state.states)+1}: '

    confidence_tokens, confidence = get_confidence_state(s_t0, 8, tokenizer, transformer_weights, model_params)
    answer = tokenizer.decode(confidence_tokens[0].tolist())
    answer = answer.strip()
    return State(states=[*state.states, SubStep(subquestion=action, subanswer=answer, confidence=confidence)],
                 prefix=state.prefix,
                 question=state.question)

def generate_with_reward():
    return
