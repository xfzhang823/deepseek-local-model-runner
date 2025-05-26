"""
/quantize/scrooge_scale.py

Custom version of the official code to apply scales to a single layer.
"""

import logging
from typing import List, Tuple, Optional, Dict, cast
import torch
import torch.nn as nn
from awq.utils.utils import get_best_device
from awq.modules.act import ScaledActivation
from awq.utils.module import get_op_by_name, set_op_by_name, get_op_name
from awq.quantize.scale import scale_fc_fc, scale_gelu_fc
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


import torch
import torch.nn as nn
from typing import Dict, List, Union


def normalize_scales_across_groups(
    layers: List[Union[nn.Linear, Tuple[nn.Linear, slice]]],
    name_to_layer: Dict[Union[nn.Linear, Tuple[nn.Linear, slice]], str],
    scales_dict: Dict[str, torch.Tensor],
    epsilon: float = 1e-6,
) -> Dict[str, torch.Tensor]:
    """
    Normalize group-wise scales across Q/K/V projections that share input.
    Supports both:
    - Separate layers (each scale is [O, G])
    - Fused layers using weight slices (e.g., query_key_value), scale sliced accordingly.

    Args:
        layers: List of nn.Linear layers or (layer, slice) for fused QKV
        name_to_layer: Dict mapping layer or (layer, slice) → name
        scales_dict: Dict name → scale tensor (shape [O, G])
        epsilon: Small constant for numerical stability
        group_size: Number of input dims per group

    Returns:
        Updated scales_dict with normalized scales
    """
    per_group_max = {}

    for entry in layers:
        name = name_to_layer[entry]

        if isinstance(entry, tuple):
            layer, sl = entry
            weight = layer.weight.detach()[sl]  # [O, I]
            scale = scales_dict[name]
        else:
            layer = entry
            weight = layer.weight.detach()  # [O, I]
            scale = scales_dict[name]

        assert scale.dim() == 2, f"{name} must be 2D [O, G], got {scale.shape}"
        O, I = weight.shape
        G = scale.shape[1]
        assert I % G == 0, f"Invalid group size: in_features={I}, groups={G}"
        gsize = I // G

        per_group_max[name] = []
        for g in range(G):
            w = weight[:, g * gsize : (g + 1) * gsize]
            s = scale[:, g].view(-1, 1)
            max_val = (w / (s + epsilon)).abs().max().item()
            per_group_max[name].append(max_val)

    # Step 2: For each group index, get shared max across layers
    G = len(next(iter(per_group_max.values())))
    shared_max = [
        max(per_group_max[name][g] for name in per_group_max) for g in range(G)
    ]

    # Step 3: Normalize each layer's scale
    for entry in layers:
        name = name_to_layer[entry]
        scale = scales_dict[name]
        current_max = per_group_max[name]

        normed = torch.empty_like(scale)
        for g in range(G):
            ratio = current_max[g] / (shared_max[g] + epsilon)
            normed[:, g] = scale[:, g] * ratio

        scales_dict[name] = normed

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
    Applies group-wise scaling to a Linear layer's weights in-place.

    This function assumes that the input dimension of the layer is divided into
    equal-sized groups (e.g., group_size = in_features // num_groups), and that
    each group has a corresponding scale per output channel.

    Args:
        layer (nn.Linear): The Linear layer whose weights will be modified.
        scales (torch.Tensor): A 2D tensor of shape [out_features, num_groups],
            where each entry scales a corresponding group of input weights.

    Behavior:
        For each group g:
            - Scales the weight slice [:, g_start:g_end] by scales[:, g]
            - Modifies layer.weight.data in-place

    Raises:
        AssertionError: If scale dimensions do not align with the layer's weight shape.
    """
    try:
        weight = layer.weight.data
        O, I = weight.shape

        if scales.dim() != 2:
            raise ValueError(
                f"Expected scales to be 2D [out_features, num_groups], got {scales.shape}"
            )

        S_O, G = scales.shape
        if O != S_O:
            raise ValueError(
                f"Mismatch: weight.out_features={O}, but scales.shape[0]={S_O}"
            )

        if I % G != 0:
            raise ValueError(
                f"Incompatible group size: in_features={I}, num_groups={G} → not divisible"
            )

        group_size = I // G
        logger.debug(
            f"[scale_weights_by_group] layer={layer.__class__.__name__}, "
            f"weight.shape={weight.shape}, scale.shape={scales.shape}, "
            f"group_size={group_size}, num_groups={G}"
        )

        for g in range(G):
            start = g * group_size
            end = (g + 1) * group_size
            s = scales[:, g].view(-1, 1)  # [O, 1]
            weight[:, start:end].mul_(s)

        logger.info(
            f"[scale_weights_by_group] Successfully applied group-wise scale: "
            f"{G} groups × {O} output neurons"
        )

    except Exception as e:
        logger.error(f"[scale_weights_by_group] Failed to apply scaling: {e}")
        raise


@torch.no_grad()
def apply_clip(
    module: nn.Module,
    layer_name: str,
    clip_value: Optional[float],
) -> None:
    """
    Apply symmetric clipping to a Linear layer's weights.

    Clipping is only applied if `clip_value` is not None.

    ---
    Operation:
        weight[:] = clamp(weight[:], -clip_value, +clip_value)

    Args:
        module (nn.Module): Parent module containing the layer.
        layer_name (str): Name of the layer (dot-path) to clip.
        clip_value (float): Max absolute value to retain in weight.
            No-op if None.

    Raises:
        TypeError: If layer is not nn.Linear.
    """
    if clip_value is None:
        return

    layer = cast(nn.Linear, get_op_by_name(module, layer_name))
    if not isinstance(layer, nn.Linear):
        raise TypeError(f"Expected nn.Linear, got {type(layer)} for layer {layer_name}")

    device = get_best_device()
    layer.to(device)

    max_val_tensor = torch.tensor(clip_value, device=device)
    original_shape = layer.weight.shape

    layer.weight.data = torch.clamp(
        layer.weight.data.view(-1), -max_val_tensor, max_val_tensor
    ).view(original_shape)

    logger.debug(f"[apply_clip] {layer_name} clipped to ±{clip_value:.4f}")
    layer.cpu()
