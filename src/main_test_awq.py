import torch
import torch.nn as nn

layer = nn.Linear(5, 3)
original_weight = layer.weight

# Detach the weights
detached_weight = layer.weight.detach()

# Modify the original weights
layer.weight.data += 1

print((layer.weight != original_weight).all())  # Output: True
print((layer.weight != detached_weight).all())  # Output: True
