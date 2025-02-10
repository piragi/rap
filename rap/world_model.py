from typing import List, NamedTuple, Tuple, Optional

import torch

from config import ModelParams
from inference import generate, get_confidence_state, get_self_eval
from tokenizer import Tokenizer
from weights import TransformerWeights
from token_tracker import TokenUsageStats

Action = str

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

class SubStep(NamedTuple):
    """Represents a single step in the reasoning process.

    Attributes:
        subquestion: The question/action for this step
        subanswer: The model's answer to the subquestion
        confidence: Model's confidence in the answer (0-1)
    """
    subquestion: Action
    subanswer: str
    confidence: float

class State(NamedTuple):
    """Represents the complete state of the reasoning process.
    
    Attributes:
        states: List of previous reasoning steps
        prefix: Prompt prefix for model interactions
        question: Original problem question
    """
    states: list[SubStep]
    prefix: str
    question: str

def build_prompt(state: State, subquestions=None, action=None):
    """Build prompt for the language model based on current state.
    
    Constructs a formatted prompt incorporating the question history and either
    generating new questions or adding a specific action.
    
    Args:
        state: Current reasoning state
        subquestions: Optional list of candidate next questions
        action: Optional specific next action
        
    Returns:
        Either a single prompt string or list of prompts for batch processing
    """
    s_t0 = f"\nQuestion 5:{state.question}"
    s_t0 += ''.join(f'\nQuestion 5.{i+1}: {substate.subquestion}\nAnswer 5.{i+1}: {substate.subanswer}' for i, substate in enumerate(state.states))

    if subquestions is not None:
        return [s_t0 + f'\nNew question 5.{len(state.states)+1}: {subq}' for subq in subquestions]
    elif action is not None:
        s_t0 += f'\nQuestion 5.{len(state.states)+1}: {action}\nAnswer 5.{len(state.states)+1}: '
        return state.prefix + s_t0

    s_t0 += f'\nQuestion 5.{len(state.states)+1}: '
    return state.prefix + s_t0

def check_ending(text):
    if '\n' in text:
        text = text.split('\n')[0]
    if '.\n' in text:
        text = text.split('.\n')[0]
    if '\n\n' in text:
        text = text.split('\n\n')[0]
    return text

def advance_tokens(prompt: List[str], tokenizer: Tokenizer, transformer_weights: TransformerWeights, model_params: ModelParams, token_stats: Optional[TokenUsageStats] = None):
    """Generate tokens for one or multiple prompts.
    
    Handles batch processing of prompts, including padding and device placement.
    
    Args:
        prompt: List of prompt strings
        tokenizer: Tokenizer for model interactions
        transformer_weights: Model weights
        model_params: Model parameters
        token_stats: Optional token usage tracker
        
    Returns:
        List of generated text responses
    """
    tokens = [tokenizer.encode(p, bos=False, eos=False, allowed_special='all') for p in prompt]
    max_len = max(len(t) for t in tokens)
    tokens = [t + [tokenizer.pad_id] * (max_len - len(t)) for t in tokens]
    tokens = torch.tensor(tokens, device=device)

    generated_tokens = generate(transformer_weights, model_params, tokens, tokenizer, track_method="generate_action", token_stats=token_stats)

    final_texts = []
    for seq in generated_tokens:
        text = tokenizer.decode(seq.tolist())
        text = check_ending(text)
        text = text.strip()
        final_texts.append(text)

    return final_texts

def predict_action(state: State, tokenizer: Tokenizer, transformer_weights: TransformerWeights, model_params: ModelParams,
                   batch_size: int, token_stats: Optional[TokenUsageStats] = None) -> List[Tuple[Action, float]]:
    """Predict possible next actions given current state.
    
    Generates multiple candidate next questions and their confidence scores.
    
    Args:
        state: Current reasoning state
        tokenizer: Tokenizer for model interactions
        transformer_weights: Model weights
        model_params: Model parameters
        batch_size: Number of candidate actions to generate
        token_stats: Optional token usage tracker
        
    Returns:
        List of (action, confidence) pairs for candidate next steps
    """
    prompt = build_prompt(state)
    prompts = [prompt] * batch_size
    subquestions = advance_tokens(prompts, tokenizer, transformer_weights, model_params, token_stats)

    a_t0s = build_prompt(state, subquestions=subquestions)
    confidences = get_self_eval(a_t0s, tokenizer, transformer_weights, model_params)

    return list(zip(subquestions, confidences))

def predict_state(state: State,
                  action: Action,
                  tokenizer: Tokenizer,
                  transformer_weights: TransformerWeights,
                  model_params: ModelParams,
                  confidence=1,
                  token_stats: Optional[TokenUsageStats] = None) -> State:
    """Generate next state by applying an action to current state.
    
    Takes an action and generates the model's response to create new state.
    
    Args:
        state: Current reasoning state
        action: Action to apply
        tokenizer: Tokenizer for model interactions
        transformer_weights: Model weights
        model_params: Model parameters
        confidence: Confidence threshold
        token_stats: Optional token usage tracker
        
    Returns:
        New state incorporating the action and model's response
    """
    s_t0 = build_prompt(state, action=action)

    confidence_tokens, confidence = get_confidence_state(s_t0, confidence, tokenizer, transformer_weights, model_params, token_stats=token_stats)
    answer = tokenizer.decode(confidence_tokens[0].tolist())
    answer = check_ending(answer)

    # Create new state
    return State(states=[*state.states, SubStep(subquestion=action, subanswer=answer.strip(), confidence=confidence)],
                 prefix=state.prefix,
                 question=state.question)
