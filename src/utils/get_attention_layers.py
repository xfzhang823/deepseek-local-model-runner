from typing import Dict, List, Tuple, Union
import torch.nn as nn
from collections import defaultdict


def get_qkv_layers_by_block(
    model: nn.Module,
) -> Dict[int, Dict[str, Tuple[str, Union[nn.Linear, Tuple[nn.Linear, slice]]]]]:
    """
    Detects Q/K/V projection layers (fused or separate) for each transformer block.

    Returns:
        A dict mapping block index → {
            "q": (name, Linear) or (name, (Linear, slice)),
            "k": ...,
            "v": ...
        }
    """
    qkv_by_block = defaultdict(dict)

    for full_name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue

        lname = full_name.lower()

        # Attempt to extract block index from path (e.g., layers.0.self_attn.q_proj)
        parts = full_name.split(".")
        block_idx = None
        for i, part in enumerate(parts):
            if part in {"layers", "h"}:
                try:
                    block_idx = int(parts[i + 1])
                    break
                except (IndexError, ValueError):
                    pass
        if block_idx is None:
            continue

        # Case 1: Separate Q, K, V projections
        if "q_proj" in lname:
            qkv_by_block[block_idx]["q"] = (full_name, module)
        elif "k_proj" in lname:
            qkv_by_block[block_idx]["k"] = (full_name, module)
        elif "v_proj" in lname:
            qkv_by_block[block_idx]["v"] = (full_name, module)

        # Case 2: Fused QKV layer (e.g., query_key_value, c_attn, qkv_proj)
        elif any(x in lname for x in {"query_key_value", "c_attn", "qkv_proj"}):
            out_dim = module.out_features
            if out_dim % 3 != 0:
                continue  # not truly fused qkv
            dim = out_dim // 3

            # Store slices for each part
            qkv_by_block[block_idx]["q"] = (full_name, (module, slice(0, dim)))
            qkv_by_block[block_idx]["k"] = (full_name, (module, slice(dim, 2 * dim)))
            qkv_by_block[block_idx]["v"] = (
                full_name,
                (module, slice(2 * dim, 3 * dim)),
            )

    return dict(qkv_by_block)
