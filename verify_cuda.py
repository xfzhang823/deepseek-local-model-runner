import torch

print(torch.version.cuda)  # Should output a CUDA version, not None
print(torch.cuda.is_available())  # Should return True
print(torch.cuda.device_count())  # Should be >= 1
print(torch.cuda.get_device_name(0))  # Should return your GPU name
