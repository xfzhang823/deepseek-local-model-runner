"""
Install the DeepSeek model

Only need to run this code once.
"""

import os
from dotenv import load_dotenv
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

load_dotenv()
token = os.getenv("HUGGING_FACE_TOKEN")
model_name = "casperhansen/deepseek-r1-distill-qwen-1.5b-awq"

# 1a) tokenizer can still come from transformers
tokenizer = AutoTokenizer.from_pretrained(
    model_name, trust_remote_code=True, use_auth_token=token
)

# 1b) but the **model** must be loaded via AWQ’s API
model = AutoAWQForCausalLM.from_quantized(
    model_name, device_map="auto", trust_remote_code=True, use_auth_token=token
)
