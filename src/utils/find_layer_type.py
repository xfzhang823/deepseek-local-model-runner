"""
find_layer_type.py

Layer type detection utilities for transformer-based models.

This module provides simple heuristics for identifying whether a group of
layer names belongs to the attention mechanism (e.g., uses rotary embeddings)
or the MLP feedforward block. This is useful for quantization, memory-aware
scheduling, or device placement decisions.

Functions:
- is_attention_layer(layer_names): Detects whether a group is part of
the attention mechanism.
- is_mlp_layer(layer_names): Detects whether a group is part of the MLP
(feedforward) block.

These matchers rely on common naming conventions used across popular open-source
LLM architectures including LLaMA, Qwen, DeepSeek, OPT, GPT-2, Falcon, and BLOOM.
They are intentionally keyword-based for broad compatibility.

Usage Example:
    >>> is_attention_layer(["model.layers.0.self_attn.q_proj"])
    True

    >>> is_mlp_layer(["model.layers.0.mlp.down_proj"])
    True
"""


def is_attention_layer(layer_names: list[str]) -> bool:
    """
    Heuristically determine whether a group of layers is part of an attention mechanism.

    Args:
        layer_names (list[str]): List of layer names from a group.

    Returns:
        bool: True if the group involves attention (i.e., needs RoPE), else False.
    """
    attention_keywords = [
        "q_proj",
        "k_proj",
        "v_proj",  # modular attention projections
        "query",
        "key",
        "value",  # fused/flexible formats
        "query_key_value",
        "c_attn",  # fused QKV blocks (e.g. GPT2, BLOOM)
        "self_attn",
        "attn",
        "attention",  # block/submodule level hints
    ]

    name_blob = " ".join(layer_names).lower()
    return any(kw in name_blob for kw in attention_keywords)


def is_mlp_layer(layer_names: list[str]) -> bool:
    """
    Heuristically determine if a group of layers belongs to an MLP block.

    Args:
        layer_names (list[str]): List of fully qualified layer names.

    Returns:
        bool: True if the group is part of an MLP (feedforward) block.
    """
    mlp_keywords = [
        "mlp",  # catches 'mlp.gate_proj', etc.
        "ffn",
        "gate_proj",
        "up_proj",
        "down_proj",
        "fc1",
        "fc2",
        "c_fc",
        "c_proj",  # GPT/BERT
        "dense_h_to_4h",
        "dense_4h_to_h",  # Falcon/BLOOM
    ]

    name_blob = " ".join(layer_names).lower()
    return any(kw in name_blob for kw in mlp_keywords)
