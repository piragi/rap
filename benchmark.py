import json
import os
from typing import Dict

from datasets import load_dataset
from tqdm import tqdm

from config import ModelParams
from main import Tokenizer, load_model_params, load_weights
from mcts import mcts
from world_model import State

def extract_answer(answer_text: str) -> float:
    """Extract numerical answer from the text."""
    try:
        if "The answer is" in answer_text:
            answer_str = answer_text.split("The answer is")[-1].strip().strip('.')
            # Remove any text after the number
            answer_str = ''.join(c for c in answer_str if c.isdigit() or c == '.')
            return float(answer_str)
    except:
        return None
    return None

def run_benchmark(dataset,
                  tokenizer: Tokenizer,
                  transformer_weights: dict,
                  model_params: ModelParams,
                  n_samples: int = None,
                  rollouts: int = 10,
                  depth_limit: int = 5,
                  action_generation: int = 5) -> Dict:
    """Run MCTS on GSM8K dataset and compute accuracy."""

    # Use subset of dataset if specified
    if n_samples:
        dataset = dataset.select(range(n_samples))

    correct = 0
    total = 0
    results = []

    prefix = """Given a question, please decompose it into sub-questions. For each sub-question, please answer it in a complete sentence, ending with "The answer is". When the original question is answerable, please start the subquestion with "Now we can answer the question: ".

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

    for example in tqdm(dataset):
        question = example['question']
        target = float(example['answer'].split('####')[-1].strip())

        # Initialize state with question
        init_state = State(states=[], prefix=prefix, question="Question 5: " + question)

        # Run MCTS
        try:
            final_state = mcts(init_state, rollouts, depth_limit, action_generation, tokenizer, transformer_weights, model_params)

            # Extract predicted answer
            if final_state and final_state.states:
                pred = extract_answer(final_state.states[-1].subanswer)
                if pred is not None:
                    # Compare prediction with target
                    is_correct = abs(pred - target) < 1e-6
                    correct += int(is_correct)

                    results.append({
                        'question': question,
                        'target': target,
                        'predicted': pred,
                        'correct': is_correct,
                        'steps': [state.subquestion + "\n" + state.subanswer for state in final_state.states]
                    })
        except Exception as e:
            print(f"Error processing question: {question}")
            print(f"Error: {str(e)}")
            continue

        total += 1

        # Print running accuracy
        print(f"\nRunning accuracy: {correct/total:.2%} ({correct}/{total})")

    accuracy = correct / total if total > 0 else 0

    return {'accuracy': accuracy, 'correct': correct, 'total': total, 'results': results}

if __name__ == "__main__":
    # Load model components
    home_dir = os.path.expanduser("~")
    model_path = os.path.join(home_dir, ".llama", "checkpoints", "Llama3.2-3B")
    model_params = load_model_params(os.path.join(model_path, "params.json"))
    transformer_weights = load_weights(os.path.join(model_path, "consolidated.00.pth"))
    tokenizer = Tokenizer(model_path=os.path.join(model_path, "tokenizer.model"))

    # Load GSM8K dataset
    dataset = load_dataset("openai/gsm8k", "main")
    test_dataset = dataset['test']

    # Run benchmark
    results = run_benchmark(
        dataset=test_dataset,
        tokenizer=tokenizer,
        transformer_weights=transformer_weights,
        model_params=model_params,
        n_samples=100,  # Set to None to run on full dataset
        rollouts=5,
        depth_limit=5,
        action_generation=5)

    # Save results
    with open('gsm8k_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nFinal accuracy: {results['accuracy']:.2%}")
