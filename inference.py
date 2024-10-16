import torch
from transformers import pipeline

model_id = "meta-llama/Llama-3.2-1B-Instruct"
pipe = pipeline(
    "text_generation", model=model_id, torch_dtype=torch.bfloat16, device_map="auto"
)

pipe("the answer to life is")
