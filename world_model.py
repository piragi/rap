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

def advance_tokens(prompt: str, tokenizer: Tokenizer, transformer_weights: TransformerWeights, model_params: ModelParams):
    tokens = tokenizer.encode(prompt, bos=False, eos=False, allowed_special='all')
    generated_tokens = generate(transformer_weights, model_params, tokens, tokenizer)
    torch.cuda.empty_cache()
    generated = tokenizer.decode(generated_tokens[0].tolist())
    generated = generated.strip()
    return generated

def predict_action(state: State, tokenizer: Tokenizer, transformer_weights: TransformerWeights, model_params: ModelParams) -> Tuple[Action, float]:
    s_t0 = state.prefix
    s_t0 += ''.join(f'\nQuestion 5.{i+1}: {substate.subquestion}\nAnswer 5.{i+1}: {substate.subanswer}' for i, substate in enumerate(state.states))
    s_t0 += f'\nQuestion 5.{len(state.states)+1}: '

    subquestion = advance_tokens(s_t0, tokenizer, transformer_weights, model_params)
    a_t0 = s_t0 + subquestion
    confidence = get_self_eval(a_t0, tokenizer, transformer_weights, model_params)
    return subquestion, confidence

def predict_state(state: State, action: Action, tokenizer: Tokenizer, transformer_weights: TransformerWeights, model_params: ModelParams) -> State:
    """Generate next state given an action"""
    s_t0 = state.prefix
    s_t0 += ''.join(f'\nQuestion 5.{i+1}: {substate.subquestion}\nAnswer 5.{i+1}: {substate.subanswer}' for i, substate in enumerate(state.states))
    s_t0 += f'\nQuestion 5.{len(state.states)+1}: {action}\nAnswer 5.{len(state.states)+1}: '

    confidence_tokens, confidence = get_confidence_state(s_t0, 8, tokenizer, transformer_weights, model_params)
    answer = tokenizer.decode(confidence_tokens[0].tolist())
    answer = answer.strip()
    return State(states=[*state.states, SubStep(subquestion=action, subanswer=answer, confidence=confidence)], prefix=state.prefix)

def generate_with_reward():
    return
