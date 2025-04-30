from transformers import Qwen2ForCausalLM

model = Qwen2ForCausalLM.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")

print(model.__class__)
# Qwen2ForCausalLM

print(model.model.__class__)
# Qwen2Model

# print(model.model.model.__class__)
