"""
Install the DeepSeek model

Only need to run this code once.
"""

import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from dotenv import load_dotenv


load_dotenv()
token = os.getenv("HUGGING_FACE_TOKEN")

model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
# or
# model_name = "DeepSeek-R1-Distill-Llama-8B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=token)
