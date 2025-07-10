"""
utils/awq_tensor_utils.py
Helper functions to help analyze tensors for quantization.
"""

from typing import Optional
import logging
import torch
import logging_config

logger = logging.getLogger(__name__)


def unpack_qzeros_4bit_int32(qzeros: torch.Tensor) -> torch.Tensor:
    """
    Unpack 4-bit unsigned zero-points stored in int32 format. Each int32 contains 8 values.

    This is often used in AWQ models to store zero-points per group in compact form.

    Args:
        qzeros (torch.Tensor): Packed zero-points, shape [*, N], dtype int32

    Returns:
        torch.Tensor: Unpacked zero-points, shape [*, N * 8], dtype int8
    """
    if qzeros.dtype != torch.int32:
        raise TypeError(f"Expected int32 tensor for qzeros, got {qzeros.dtype}")

    logger.debug(f"Unpacking qzeros with shape {qzeros.shape}")

    unpacked = torch.stack(
        [((qzeros >> (4 * i)) & 0xF).to(torch.int8) for i in range(8)], dim=-1
    )  # shape: [..., 8]

    unpacked = unpacked.view(*qzeros.shape[:-1], -1)
    logger.debug(f"Unpacked qzeros shape: {unpacked.shape}")

    return unpacked


def unpack_qweight_4bit_int32(qweight: torch.Tensor) -> torch.Tensor:
    """
    Unpack 4-bit signed quantized weights from int32 format. Each int32 stores 8 values.
    Encoded using 2's complement: range is [-8, +7].

    Args:
        qweight (torch.Tensor): Packed quantized weights, shape [..., N], dtype int32

    Returns:
        torch.Tensor: Unpacked weights, shape [..., N * 8], dtype int8
    """
    if qweight.dtype != torch.int32:
        raise TypeError(f"Expected int32 tensor for qweight, got {qweight.dtype}")

    logger.debug(f"Unpacking qweight with shape {qweight.shape}")

    unpacked = torch.stack(
        [((qweight >> (4 * i)) & 0xF).to(torch.int8) for i in range(8)], dim=-1
    )  # shape: [..., 8]

    # Convert unsigned 4-bit to signed int8 using 2's complement
    unpacked = torch.where(unpacked >= 8, unpacked - 16, unpacked)

    unpacked = unpacked.view(*qweight.shape[:-1], -1)
    logger.debug(f"Unpacked qweight shape: {unpacked.shape}")

    return unpacked


def dequantize_awq_weights(
    qweight: torch.Tensor,
    scales: torch.Tensor,
    group_size: int,
    qzeros: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Dequantize AWQ-style quantized weights, with optional per-group zero-points.

    This function reconstructs float32 weights using:
        W_fp32 = (Q - Z) * S  if qzeros is provided
        W_fp32 = Q * S        if qzeros is None

    Args:
        qweight (torch.Tensor): Unpacked quantized weights [O, I], dtype torch.int8
        scales (torch.Tensor): Per-group scales [O, G], where G = I // group_size
        group_size (int): Number of input features per group (e.g., 128)
        qzeros (torch.Tensor, optional): Unpacked zero points [O, I],
            dtype torch.int8 or torch.uint8

    Returns:
        torch.Tensor: Dequantized weights [O, I], dtype torch.float32

    Raises:
        ValueError: If input shapes are incompatible.

    Examples:
        >>> qweight = torch.randint(-8, 8, (4, 16), dtype=torch.int8)
        >>> scales = torch.rand(4, 4) * 0.05  # for group_size = 4
        >>> w_fp32 = dequantize_awq_weights(qweight, scales, group_size=4)
    """
    if qweight.dtype != torch.int8:
        raise TypeError(f"Expected int8 tensor for qweight, got {qweight.dtype}")
    if scales.dtype not in (torch.float16, torch.float32):
        raise TypeError(f"Expected float16 or float32 for scales, got {scales.dtype}")
    if qweight.ndim != 2 or scales.ndim != 2:
        raise ValueError("qweight and scales must be 2D tensors")

    O, I = qweight.shape  # Output dim, Input dim
    G = I // group_size  # Group size

    if scales.shape != (O, G):
        raise ValueError(
            f"scales shape {scales.shape} does not match expected ({O}, {G})"
        )

    if qzeros is not None:
        if qzeros.shape != (O, G):
            raise ValueError(
                f"qzeros shape {qzeros.shape} does not match qweight shape {qweight.shape}"
            )
        if qzeros.dtype not in (torch.int8, torch.uint8):
            raise TypeError(f"Expected int8 or uint8 for qzeros, got {qzeros.dtype}")

    w_fp32 = torch.empty_like(qweight, dtype=torch.float32)

    for g in range(G):
        start = g * group_size
        end = (g + 1) * group_size
        scale = scales[:, g].unsqueeze(1)  # [O, 1]

        q_slice = qweight[:, start:end].float()
        if qzeros is not None:
            zero = qzeros[:, g].unsqueeze(1)  # [O, 1]
            q_slice = q_slice - zero

        w_fp32[:, start:end] = q_slice * scale

    return w_fp32


def load_tensor_from_safetensors(path: str, tensor_name: str) -> torch.Tensor:
    """
    Load a specific tensor by name from a .safetensors file.

    Args:
        path (str): Path to the .safetensors file.
        tensor_name (str): Name of the tensor inside the file
            (e.g. 'model.layers.0.self_attn.q_proj.qweight').

    Returns:
        torch.Tensor: Loaded tensor.
    """
    import safetensors.torch as st

    tensors = st.load_file(path)
    if tensor_name not in tensors:
        raise KeyError(f"Tensor '{tensor_name}' not found in {path}")
    return tensors[tensor_name]


def load_tensor_from_pt(path: str, tensor_name: str) -> torch.Tensor:
    """
    Load a specific tensor by name from a PyTorch .pt or .bin file containing a state_dict.

    Args:
        path (str): Path to the file (e.g., "model.layers.0.self_attn.q_proj.pt").
        tensor_name (str): Key in the state_dict (e.g., "qweight").

    Returns:
        torch.Tensor: The requested tensor.
    """
    obj = torch.load(path, map_location="cpu")
    if not isinstance(obj, dict):
        raise TypeError(f"Expected a dict in {path}, got {type(obj)}")
    if tensor_name not in obj:
        raise KeyError(f"'{tensor_name}' not found in {path}")
    return obj[tensor_name]
