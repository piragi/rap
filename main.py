import os

from config import load_model_params
from tokenizer import Tokenizer
from weights import load_weights

home_dir = os.path.expanduser("~")
model_path = os.path.join(home_dir, ".llama", "checkpoints", "Llama3.2-1B-Instruct")

model_params = load_model_params(os.path.join(model_path, "params.json"))
transformer_weights = load_weights(os.path.join(model_path, "consolidated.00.pth"))
tokenizer = Tokenizer(model_path=os.path.join(model_path, "tokenizer.model"))
