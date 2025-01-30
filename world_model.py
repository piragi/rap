from typing import List, NamedTuple, Tuple, Union
import torch
from config import ModelParams
from main import generate, get_confidence_state, get_self_eval
from tokenizer import Tokenizer
from weights import TransformerWeights

Action = str

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SubStep(NamedTuple):
    """Represents a single step in the reasoning process"""
    subquestion: Action
    subanswer: str
    confidence: float

class State(NamedTuple):
    """Represents the complete state of the reasoning process"""
    states: list[SubStep]
    prefix: str
    question: str

def clean_generated_text(text: str) -> str:
    """Clean up generated text by removing unwanted line breaks and extra whitespace"""
    for separator in ['\n', '.\n', '\n\n']:
        if separator in text:
            text = text.split(separator)[0]
    return text.strip()

def build_prompt(state: State, action: str, is_answer: bool = False, include_prefix: bool = True) -> str:
    """Build prompt for the current state and action"""
    base = f"\nQuestion 5: {state.question}"
    states_text = ''.join(
        f'\nQuestion 5.{i+1}: {substate.subquestion}\nAnswer 5.{i+1}: {substate.subanswer}'
        for i, substate in enumerate(state.states)
    )
    if is_answer:
        next_state = f'\nQuestion 5.{len(state.states)+1}: {action}\nAnswer 5.{len(state.states)+1}: '
    else:
        next_state = f'\nQuestion 5.{len(state.states)+1}: '
    
    content = base + states_text + (next_state if action else '')
    return state.prefix + content if include_prefix else content

def advance_tokens(prompt: List[str], tokenizer: Tokenizer, 
                  transformer_weights: TransformerWeights, 
                  model_params: ModelParams) -> List[str]:
    """Generate tokens for one or multiple prompts."""
    tokens = [tokenizer.encode(p, bos=False, eos=False, allowed_special='all') 
             for p in prompt]
    
    max_len = max(len(t) for t in tokens)
    tokens = [t + [tokenizer.pad_id] * (max_len - len(t)) for t in tokens]
    tokens = torch.tensor(tokens, device=DEVICE)

    generated_tokens = generate(transformer_weights, model_params, tokens, tokenizer)
    return [clean_generated_text(tokenizer.decode(seq.tolist())) 
            for seq in generated_tokens]

def predict_action(state: State, tokenizer: Tokenizer, 
                  transformer_weights: TransformerWeights,
                  model_params: ModelParams, 
                  batch_size: int) -> List[Tuple[Action, float]]:
    """Predict next action(s) given current state."""
    base_prompt = build_prompt(state, '', include_prefix=True)
    prompts = [base_prompt] * batch_size
    subquestions = advance_tokens(prompts, tokenizer, transformer_weights, model_params)
    
    base_prompt_no_prefix = build_prompt(state, '', include_prefix=False)
    eval_prompts = [base_prompt_no_prefix + f'\nNew question 5.{len(state.states)+1}: {subq}' 
                    for subq in subquestions]
    
    confidences = get_self_eval(eval_prompts, tokenizer, transformer_weights, model_params)
    return list(zip(subquestions, confidences))

def predict_state(state: State, action: Action, 
                 tokenizer: Tokenizer,
                 transformer_weights: TransformerWeights,
                 model_params: ModelParams, 
                 confidence: int = 1) -> State:
    """Generate next state given an action"""
    prompt = build_prompt(state, action, is_answer=True)
    
    confidence_tokens, conf_value = get_confidence_state(
        prompt, confidence, tokenizer, transformer_weights, model_params
    )
    
    answer = clean_generated_text(tokenizer.decode(confidence_tokens[0].tolist()))
    
    return State(
        states=[*state.states, SubStep(
            subquestion=action,
            subanswer=answer,
            confidence=conf_value
        )],
        prefix=state.prefix,
        question=state.question
    )
