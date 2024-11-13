from typing import NamedTuple

import torch

from config import ModelParams
from main import generate
from tokenizer import Tokenizer
from weights import TransformerWeights

class SubStep(NamedTuple):
    subquestion: str
    subanswer: str
    confidence: float

class WorldModel(NamedTuple):
    states: list[SubStep]
    prefix: str

def predict_state(world_model: WorldModel, tokenizer: Tokenizer, transformer_weights: TransformerWeights, model_params: ModelParams) -> WorldModel:
    s_t = world_model.prefix
    s_t += ''.join(f'\nQuestion 5.{i+1}: {state.subquestion}\nAnswer 5.{i+1}: {state.subanswer}' for i, state in enumerate(world_model.states))

    s_t += f'\nQuestion 5.{len(world_model.states)+1}:'
    tokens = tokenizer.encode(s_t, bos=True, eos=False, allowed_special='all')
    subquestion_tokens = generate(transformer_weights, model_params, tokens, tokenizer)
    subquestion = tokenizer.decode(subquestion_tokens[0].tolist())
    subquestion = subquestion.strip()
    s_t += subquestion

    s_t += f'\nAnswer 5.{len(world_model.states)+1}: '
    tokens = tokenizer.encode(s_t, bos=True, eos=False, allowed_special='all')
    subanswer_tokens = generate(transformer_weights, model_params, tokens, tokenizer)
    subanswer = tokenizer.decode(subanswer_tokens[0].tolist())
    subanswer = subanswer.strip()

    world_model.states.append(SubStep(subquestion=subquestion, subanswer=subanswer, confidence=0))
    return world_model
