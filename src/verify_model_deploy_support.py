from transformers import AutoConfig
import awq

config = AutoConfig.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
print(config.architectures)  # Should include "QwenForCausalLM" or similar

from awq import AutoAWQForCausalLM  # No error = success

print("AutoAWQ works!")
