"""
/quantize/scrooge_scale.py

Custom version of the official code to apply scales to a single layer.
"""

import re
import logging
from typing import List, Tuple, Optional, Dict, cast, Union
import torch
import torch.nn as nn
from awq.utils.utils import get_best_device
from awq.utils.module import get_op_by_name
from transformers.models.bloom.modeling_bloom import BloomGelu
from transformers.models.llama.modeling_llama import LlamaRMSNorm
from transformers.models.gemma.modeling_gemma import GemmaRMSNorm
from transformers.models.gemma2.modeling_gemma2 import Gemma2RMSNorm
from transformers.models.cohere.modeling_cohere import CohereLayerNorm
from transformers.activations import NewGELUActivation, PytorchGELUTanh, GELUActivation

logger = logging.getLogger(__name__)

allowed_norms = [
    nn.LayerNorm,
    LlamaRMSNorm,
    GemmaRMSNorm,
    Gemma2RMSNorm,
    CohereLayerNorm,
]
allowed_act_fns = [
    nn.GELU,
    BloomGelu,
    NewGELUActivation,
    PytorchGELUTanh,
    GELUActivation,
]


def normalize_scales_across_groups(
    layers: List[Union[nn.Linear, Tuple[nn.Linear, slice]]],
    name_to_layer: Dict[Union[nn.Linear, Tuple[nn.Linear, slice]], str],
    scales_dict: Dict[str, torch.Tensor],
    epsilon: float = 1e-6,
) -> Dict[str, torch.Tensor]:
    """
    Normalize group-wise or per-channel-per-group scales across layers that share input.

    Supports:
        - Per-group: shape [G]
        - Per-channel-per-group: shape [O, G]

    For each group g, this ensures the scale across layers is normalized relative to
    the shared maximum value — either globally (per-group) or per output channel.

    Args:
        layers: List of nn.Linear layers or (layer, slice) pairs.
        name_to_layer: Mapping from each entry in `layers` to a unique string name.
        scales_dict: Dict mapping layer names to scale tensors.
        epsilon: Small value added to denominator for stability.

    Returns:
        Updated scales_dict with normalized values (same shape as input).
    """
    layer_names = [name_to_layer[l] for l in layers]
    sample_scale = scales_dict[layer_names[0]]
    scale_dim = sample_scale.dim()

    # Validate all layers have same shape
    for name in layer_names:
        assert (
            scales_dict[name].shape == sample_scale.shape
        ), f"{name} scale shape mismatch: expected {sample_scale.shape}, got {scales_dict[name].shape}"

    if scale_dim == 1:
        # --- Per-group [G] ---
        stacked = torch.stack(
            [scales_dict[name] for name in layer_names], dim=0
        )  # [L, G]
        shared_max = stacked.abs().amax(dim=0, keepdim=False)  # [G]
        for i, name in enumerate(layer_names):
            normed = stacked[i] / (shared_max + epsilon)  # [G]
            if torch.isnan(normed).any():
                logger.warning(f"[NaN Detected] in scale norm for layer {name}")
            scales_dict[name] = normed.clamp(min=1e-4)

    elif scale_dim == 2:
        # --- Per-channel-per-group [O, G] ---
        stacked = torch.stack(
            [scales_dict[name] for name in layer_names], dim=0
        )  # [L, O, G]
        shared_max = stacked.abs().amax(dim=0, keepdim=False)  # [O, G]
        for i, name in enumerate(layer_names):
            normed = stacked[i] / (shared_max + epsilon)  # [O, G]
            if torch.isnan(normed).any():
                logger.warning(f"[NaN Detected] in scale norm for layer {name}")
            scales_dict[name] = normed.clamp(min=1e-4, max=10.0)

    else:
        raise ValueError(
            f"Unsupported scale shape: {sample_scale.shape}. Must be 1D or 2D."
        )

    logger.debug(f"[normalize_scales] mode: {scale_dim}D, shape: {sample_scale.shape}")
    logger.debug(
        f"[normalize_scales] first layer preview: {scales_dict[layer_names[0]][:5]}"
    )

    return scales_dict


def scale_ln_fc(fc_layer: nn.Linear, scales: torch.Tensor) -> None:
    """
    * Modified from standard code in scale.py

    Scales a Linear layer's weights and biases following a normalization layer.

    Args:
        fc_layer (nn.Linear): The Linear layer to scale (in-place modification).
        scales (torch.Tensor): Per-channel scaling factors (1D tensor of shape `(out_features,)`).

    Returns:
        None
    """
    device = fc_layer.weight.device

    if scales.shape[0] != fc_layer.weight.shape[0]:
        raise ValueError(
            f"Mismatched scale size: scales={scales.shape}, "
            f"fc_layer.out_features={fc_layer.weight.shape[0]}"
        )

    scale = scales.to(device).view(1, -1)

    logger.debug(
        f"[scale_ln_fc] LayerType={fc_layer.__class__.__name__} | "
        f"scale.shape={scales.shape} | weight.shape={fc_layer.weight.shape}"
    )

    fc_layer.weight.data *= scale
    if fc_layer.bias is not None:
        fc_layer.bias.data *= scale.squeeze()


@torch.no_grad()
def apply_scale_all_groups(layer: nn.Linear, scales: torch.Tensor) -> None:
    """
    Applies per-input-channel group-wise scaling to a Linear layer's weights in-place.

    Args:
        layer (nn.Linear): Target Linear layer.
        scales (Tensor): [num_groups, group_size], scale for each input channel in group.
    """
    weight = layer.weight.data  # [out_features, in_features]
    out_features, in_features = weight.shape
    num_groups, group_size = scales.shape
    assert in_features == num_groups * group_size

    scales = scales.to(weight.device)

    for g in range(num_groups):
        start = g * group_size
        end = (g + 1) * group_size
        # scales[g]: [group_size], want to apply to all output channels, so broadcast to [1, group_size]
        weight[:, start:end].mul_(scales[g].view(1, -1))

    logger.info(
        f"[apply_scale_all_groups] Applied per-input-channel scaling: {num_groups} groups × {group_size} inputs each"
    )


@torch.no_grad()
def apply_clip(
    module: nn.Module,
    layer_name: str,
    clip_value: Optional[float],
) -> None:
    """
    Apply symmetric clipping to a Linear layer's weights.
    If layer_name is a global name, this auto-strips to relative.

    Args:
        module (nn.Module): Parent module containing the layer.
        layer_name (str): Dot-path (relative or global).
        clip_value (float): Max absolute value to retain in weight.
    """
    if clip_value is None:
        return

    # Convert global path to relative path (safe fallback)
    for candidate_prefix in ["model.layers.", "layers.", "transformer.layers."]:
        match = re.match(rf"^{re.escape(candidate_prefix)}\d+\.(.+)", layer_name)
        if match:
            original = layer_name
            layer_name = match.group(1)
            logger.debug(
                f"[apply_clip] Converted global → relative: {original} → {layer_name}"
            )
            break

    try:
        layer = cast(nn.Linear, get_op_by_name(module, layer_name))
    except ValueError as e:
        raise ValueError(
            f"❌ Failed to locate {layer_name} inside {module.__class__.__name__}"
        ) from e

    if not isinstance(layer, nn.Linear):
        raise TypeError(f"Expected nn.Linear, got {type(layer)} for {layer_name}")

    # Move to GPU, apply clamp, move back
    device = get_best_device()
    layer.to(device)

    max_val_tensor = torch.tensor(clip_value, device=device)
    layer.weight.data.clamp_(-max_val_tensor, max_val_tensor)

    if layer.bias is not None:
        layer.bias.data.clamp_(-max_val_tensor, max_val_tensor)

    logger.debug(f"[apply_clip] {layer_name} clipped to ±{clip_value:.4f}")
    layer.cpu()
