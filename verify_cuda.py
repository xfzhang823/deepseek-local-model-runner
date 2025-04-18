"""tool to verify cuda, cuda/torch compatibility"""

import torch

print(torch.__version__)  # Should be >=2.0.0

print(torch.version.cuda)  # Should output a CUDA version, not None
print(f"torch.cuda.is_available: {torch.cuda.is_available()}")  # Should return True
print(torch.cuda.device_count())  # Should be >= 1
print(torch.cuda.get_device_name(0))  # Should return your GPU name
