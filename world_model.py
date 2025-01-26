from typing import List, NamedTuple, Tuple, Union

import torch

from config import ModelParams
from main import generate, get_confidence_state, get_self_eval
from tokenizer import Tokenizer
from weights import TransformerWeights

Action = str

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

class SubStep(NamedTuple):
    subquestion: Action
    subanswer: str
    confidence: float

class State(NamedTuple):
    states: list[SubStep]
    prefix: str
    question: str

def advance_tokens(prompt: List[str], tokenizer: Tokenizer, transformer_weights: TransformerWeights, model_params: ModelParams):
    """
    Generate tokens for one or multiple prompts.
    """
    tokens = [tokenizer.encode(p, bos=False, eos=False, allowed_special='all') for p in prompt]
    max_len = max(len(t) for t in tokens)
    tokens = [t + [tokenizer.pad_id] * (max_len - len(t)) for t in tokens]
    tokens = torch.tensor(tokens, device=device)

    generated_tokens = generate(transformer_weights, model_params, tokens, tokenizer)

    final_texts = []
    for seq in generated_tokens:
        text = tokenizer.decode(seq.tolist())
        if '\n' in text:
            text = text.split('\n')[0]
        if '.\n' in text:
            text = text.split('.\n')[0]
        if '\n\n' in text:
            text = text.split('\n\n')[0]
        text = text.strip()
        final_texts.append(text)

    return final_texts

def predict_action(state: State, tokenizer: Tokenizer, transformer_weights: TransformerWeights, model_params: ModelParams,
                   batch_size: int) -> List[Tuple[Action, float]]:
    """
    Predict next action(s) given current state.
    """
    s_t0 = f"\n{state.question}"
    if "Question 1:" in state.prefix:
        s_t0 += ''.join(f'\nQuestion 5.{i+1}: {substate.subquestion}\nAnswer 5.{i+1}: {substate.subanswer}' for i, substate in enumerate(state.states))
        action = f'\nQuestion 5.{len(state.states)+1}: '
    else:
        s_t0 += ''.join(f'\nQuestion: {substate.subquestion}\nAnswer: {substate.subanswer}' for i, substate in enumerate(state.states))
        action = f'\nQuestion: '

    # Build prompt

    # Batch prediction
    prompts = [state.prefix + s_t0 + action] * batch_size
    subquestions = advance_tokens(prompts, tokenizer, transformer_weights, model_params)

    # Batch evaluate
    if "Question 1:" in state.prefix:
        a_t0s = [s_t0 + f'\nNew question 5.{len(state.states)+1}: {subq}' for subq in subquestions]
    else: 
        a_t0s = [s_t0 + f'\nNew question: {subq}' for subq in subquestions]
    confidences = get_self_eval(a_t0s, tokenizer, transformer_weights, model_params)

    return list(zip(subquestions, confidences))

def predict_state(state: State, action: Action, tokenizer: Tokenizer, transformer_weights: TransformerWeights, model_params: ModelParams, confidence=8) -> State:
    """Generate next state given an action"""
    # Build prompt
    s_t0 = state.prefix + f"\n{state.question}"
    if "Question 1:" in state.prefix:
        s_t0 += ''.join(f'\nQuestion 5.{i+1}: {substate.subquestion}\nAnswer 5.{i+1}: {substate.subanswer}' for i, substate in enumerate(state.states))
        s_t0 += f'\nQuestion 5.{len(state.states)+1}: {action}\nAnswer 5.{len(state.states)+1}: '
    else:
        s_t0 += ''.join(f'\nQuestion: {substate.subquestion}\nAnswer: {substate.subanswer}' for i, substate in enumerate(state.states))
        s_t0 += f'\nQuestion: {action}\nAnswer: '

    # Get confident answer
    confidence_tokens, confidence = get_confidence_state(s_t0, confidence, tokenizer, transformer_weights, model_params)
    answer = tokenizer.decode(confidence_tokens[0].tolist())
    if '\n' in answer:
        answer = answer.split('\n')[0]
    if '.\n' in answer:
        answer = answer.split('.\n')[0]
    if '\n\n' in answer:
        answer = answer.split('\n\n')[0]

    # Create new state
    return State(states=[*state.states, SubStep(subquestion=action, subanswer=answer.strip(), confidence=confidence)],
                 prefix=state.prefix,
                 question=state.question)
