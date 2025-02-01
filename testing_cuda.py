import torch

print(torch.cuda.is_available())
print(torch.cuda.device_count())

torch.cuda.empty_cache()
torch.cuda.reset_max_memory_allocated()

device = torch.cuda.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
