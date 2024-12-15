import os
import time

from config import ModelParams
from main import (Tokenizer, get_action_loglikelihood, get_confidence_state, get_self_eval, load_model_params, load_weights)
from mcts import mcts
from weights import TransformerWeights
from world_model import State, SubStep, predict_action, predict_state

PREFIX = """Given a question, please decompose it into sub-questions. For each sub-question, please answer it in a complete sentence, ending with "The answer is". When the original question is answerable, please start the subquestion with "Now we can answer the question: ".

Here are some examples: Question 1: Four years ago, Kody was only half as old as Mohamed. If Mohamed is currently twice as 30 years old, how old is Kody?
Question 1.1: How old is Mohamed?
Answer 1.1: He is currently 30 * 2 = 60 years old. The answer is 60.
Question 1.2: How old was Mohamed four years ago?
Answer 1.2: Four years ago, he must have been 60 - 4 = 56 years old. The answer is 56.
Question 1.3: How old was Kody four years ago?
Answer 1.3: Kody was half as old as Mohamed four years ago. Thus, Kody was 56 / 2 = 28 years old. The answer is 28.
Question 1.4: Now we can answer the question: How old is Kody?
Answer 1.4: She is currently 28 + 4 = 32 years old. The answer is 32.

Question 2: On a moonless night, three fireflies danced in the evening breeze. They were joined by four less than a dozen more fireflies before two of the fireflies flew away. How many fireflies remained?
Question 2.1: How many fireflies joined?
Answer 2.1: The fireflies were joined by four less than a dozen more fireflies, which are 12 - 4 = 8 fireflies. The answer is 8.
Question 2.2: Now we can answer the question: How many fireflies remained?
Answer 2.2: Three fireflies were dancing originally. They were joined by 8 fireflies before two of them flew away. So there were 3 + 8 - 2 = 9 remaining. The answer is 9.

Question 3: Ali has four $10 bills and six $20 bills that he saved after working for Mr. James on his farm. Ali gives her sister half of the total money he has and uses 3/5 of the remaining amount of money to buy dinner. Calculate the amount of money he has after buying the dinner.
Question 3.1: How much money does Ali have in total?
Answer 3.1: Ali has four $10 bills and six $20 bills. So he has 4 * 10 + 6 * 20 = 160 dollars. The answer is 160.
Question 3.2: How much money does Ali give to his sister?
Answer 3.2: Ali gives half of the total money he has to his sister. So he gives 160 / 2 = 80 dollars to his sister. The answer is 80.
Question 3.3: How much money does Ali have after giving his sister the money?
Answer 3.3: After giving his sister the money, Ali has 160 - 80 = 80 dollars left. The answer is 80.
Question 3.4: How much money does Ali use to buy dinner?
Answer 3.4: Ali uses 3/5 of the remaining amount of money to buy dinner. So he uses 80 * 3/5 = 48 dollars to buy dinner. The answer is 48.
Question 3.5: Now we can answer the question: How much money does Ali have after buying the dinner?
Answer 3.5: After buying the dinner, Ali has 80 - 48 = 32 dollars left. The answer is 32.

Question 4: A car is driving through a tunnel with many turns. After a while, the car must travel through a ring that requires a total of 4 right-hand turns. After the 1st turn, it travels 5 meters. After the 2nd turn, it travels 8 meters. After the 3rd turn, it travels a little further and at the 4th turn, it immediately exits the tunnel. If the car has driven a total of 23 meters around the ring, how far did it have to travel after the 3rd turn?
Question 4.1: How far did the car travel except for the 3rd turn?
Answer 4.1: It travels 5 meters after the 1st, 8 meters after the 2nd, and 0 meters after the 4th turn. It’s a total of 5 + 8 + 0 = 13 meters. The answer is 13.
Question 4.2: Now we can answer the question: How far did the car have to travel after the 3rd turn?
Answer 4.2: The car has driven a total of 23 meters around the ring. It travels 13 meters except for the 3rd turn. So it has to travel 23 - 13 = 10 meters after the 3rd turn. The answer is 10."""

QUESTION = "Question 5: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?"

PREFIX2 = """Given a question, please decompose it into sub-questions. For each sub-question, please answer it in a complete sentence, ending with "The answer is". When the original question is answerable, please start the subquestion with "Now we can answer the question: ".

Here are some examples: Question 1: Four years ago, Kody was only half as old as Mohamed. If Mohamed is currently twice as 30 years old, how old is Kody?
Question 1.1: How old is Mohamed?
Answer 1.1: He is currently 30 * 2 = 60 years old. The answer is 60.
Question 1.2: How old was Mohamed four years ago?
Answer 1.2: Four years ago, he must have been 60 - 4 = 56 years old. The answer is 56.
Question 1.3: How old was Kody four years ago?
Answer 1.3: Kody was half as old as Mohamed four years ago. Thus, Kody was 56 / 2 = 28 years old. The answer is 28.
Question 1.4: Now we can answer the question: How old is Kody?
Answer 1.4: She is currently 28 + 4 = 32 years old. The answer is 32.

Question 2: On a moonless night, three fireflies danced in the evening breeze. They were joined by four less than a dozen more fireflies before two of the fireflies flew away. How many fireflies remained?
Question 2.1: How many fireflies joined?
Answer 2.1: The fireflies were joined by four less than a dozen more fireflies, which are 12 - 4 = 8 fireflies. The answer is 8.
Question 2.2: Now we can answer the question: How many fireflies remained?
Answer 2.2: Three fireflies were dancing originally. They were joined by 8 fireflies before two of them flew away. So there were 3 + 8 - 2 = 9 remaining. The answer is 9.

Question 3: Ali has four $10 bills and six $20 bills that he saved after working for Mr. James on his farm. Ali gives her sister half of the total money he has and uses 3/5 of the remaining amount of money to buy dinner. Calculate the amount of money he has after buying the dinner.
Question 3.1: How much money does Ali have in total?
Answer 3.1: Ali has four $10 bills and six $20 bills. So he has 4 * 10 + 6 * 20 = 160 dollars. The answer is 160.
Question 3.2: How much money does Ali give to his sister?
Answer 3.2: Ali gives half of the total money he has to his sister. So he gives 160 / 2 = 80 dollars to his sister. The answer is 80.
Question 3.3: How much money does Ali have after giving his sister the money?
Answer 3.3: After giving his sister the money, Ali has 160 - 80 = 80 dollars left. The answer is 80.
Question 3.4: How much money does Ali use to buy dinner?
Answer 3.4: Ali uses 3/5 of the remaining amount of money to buy dinner. So he uses 80 * 3/5 = 48 dollars to buy dinner. The answer is 48.
Question 3.5: Now we can answer the question: How much money does Ali have after buying the dinner?
Answer 3.5: After buying the dinner, Ali has 80 - 48 = 32 dollars left. The answer is 32.

Question 4: A car is driving through a tunnel with many turns. After a while, the car must travel through a ring that requires a total of 4 right-hand turns. After the 1st turn, it travels 5 meters. After the 2nd turn, it travels 8 meters. After the 3rd turn, it travels a little further and at the 4th turn, it immediately exits the tunnel. If the car has driven a total of 23 meters around the ring, how far did it have to travel after the 3rd turn?
Question 4.1: How far did the car travel except for the 3rd turn?
Answer 4.1: It travels 5 meters after the 1st, 8 meters after the 2nd, and 0 meters after the 4th turn. It’s a total of 5 + 8 + 0 = 13 meters. The answer is 13.
Question 4.2: Now we can answer the question: How far did the car have to travel after the 3rd turn?
Answer 4.2: The car has driven a total of 23 meters around the ring. It travels 13 meters except for the 3rd turn. So it has to travel 23 - 13 = 10 meters after the 3rd turn. The answer is 10."""

QUESTION2 = "Question 5: Melanie is a door-to-door saleswoman. She sold a third of her vacuum cleaners at the green house, 2 more to the red house, and half of what was left at the orange house. If Melanie has 5 vacuum cleaners left, how many did she start with?"

def test_log_likelihood(tokenizer: Tokenizer, transformer_weights: TransformerWeights, model_params: ModelParams):
    prefix1 = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are only responding with what you are asked for.<|eot_id|><|start_header_id|>user<|end_header_id|>

Answer only with "Yes we can".<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""
    actions1 = ["Yes we can", "No we don't"]
    probs1 = get_action_loglikelihood(prefix1, actions1, tokenizer, transformer_weights, model_params)
    assert probs1[0] == 0.0, "probability for expected action not 0.0"

def test_confidence_state(tokenizer: Tokenizer, transformer_weights: TransformerWeights, model_params: ModelParams):
    prefix = PREFIX + f"\n\n{QUESTION}" + "\nQuestion 5.1: How many clips did Natalia sell in May?"
    most_confident_state, _ = get_confidence_state(prefix, 8, tokenizer, transformer_weights, model_params)
    answer = tokenizer.decode(most_confident_state[0].tolist())
    assert "The answer is 24." in answer, answer

def test_self_eval(tokenizer: Tokenizer, transformer_weights: TransformerWeights, model_params: ModelParams):
    prefix_correct = """Question 5: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?"""
    subquestion_correct = "How many clips did Natalia sell in May?"
    prefix_incorrect = """Question 5: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?"""
    subquestion_incorrect = "What is the double of 48?"
    prob_correct = get_self_eval(prefix_correct, subquestion_correct, tokenizer, transformer_weights, model_params)
    prob_incorrect = get_self_eval(prefix_incorrect, subquestion_incorrect, tokenizer, transformer_weights, model_params)
    print(f'YES: {prob_correct}, NO: {prob_incorrect} ')
    assert prob_correct > prob_incorrect, f'Probability over "Yes" token of correct resoning not larger than probability of incorrect reasoning: {prob_correct} < {prob_incorrect}'

def test_predict_state(tokenizer: Tokenizer, transformer_weights: TransformerWeights, model_params: ModelParams):
    #TODO: there are still problems with extracting the answer, when additional tokens are produced
    state = State(states=[], prefix=PREFIX, question=QUESTION)
    steps = 5
    last_state = SubStep
    for _ in range(steps):
        action, _ = predict_action(state, tokenizer, transformer_weights, model_params)
        state = predict_state(state, action, tokenizer, transformer_weights, model_params)
        last_state = state.states[-1]
        if last_state.subquestion.startswith("Now we can answer the question"): break

    assert last_state.subanswer.endswith("The answer is 72."), last_state.subanswer

def test_mcts_iteration(tokenizer: Tokenizer, transformer_weights: TransformerWeights, model_params: ModelParams):
    start = time.time()
    state = State(states=[], prefix=PREFIX, question=QUESTION)
    best_state = mcts(state, 10, 5, 1, tokenizer, transformer_weights, model_params)
    answer = best_state.states[-1].subanswer
    end = time.time()
    assert answer.endswith("The answer is 72."), answer
    print(f'time for one iteration: {end - start}')

if __name__ == "__main__":
    print("Running tests...")
    # Load model components
    home_dir = os.path.expanduser("~")
    model_path = os.path.join(home_dir, ".llama", "checkpoints", "Llama3.2-3B")
    model_params = load_model_params(os.path.join(model_path, "params.json"))
    transformer_weights = load_weights(os.path.join(model_path, "consolidated.00.pth"))
    tokenizer = Tokenizer(model_path=os.path.join(model_path, "tokenizer.model"))

    # test_log_likelihood(tokenizer, transformer_weights, model_params)
    # test_confidence_state(tokenizer, transformer_weights, model_params)
    # test_self_eval(tokenizer, transformer_weights, model_params)
    # test_predict_state(tokenizer, transformer_weights, model_params)
    test_mcts_iteration(tokenizer, transformer_weights, model_params)
