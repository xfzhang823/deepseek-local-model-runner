import os
from dotenv import load_dotenv
from transformers import AutoConfig, AutoModelForCausalLM
from awq import AutoAWQForCausalLM  # No error = success
import awq
from safetensors.torch import load_file


model_name = os.getenv("MODEL_NAME")

config = AutoConfig.from_pretrained(model_name)
print(config.architectures)  # Should include "QwenForCausalLM" or similar

model = AutoModelForCausalLM.from_pretrained(model_name)
print(f"{model_name} parameters (w/t model.parameters): ")
print(next(model.parameters()).dtype)

# Replace this with your actual blob path
path = "/home/xzhang/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B/blobs/58858233513d76b8703e72eed6ce16807b523328188e13329257fb9594462945"
state_dict = load_file(path)

print(f"{model_name} parameters (with safetensors): ")
# Print dtype of a few parameters
for name, tensor in state_dict.items():
    print(f"{name}: {tensor.dtype}")
    break  # just one is enough to know

print("AutoAWQ works!")
