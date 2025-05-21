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


def normalize_scales_across_group(
    layers: List[nn.Linear],
    name_to_layer: Dict[nn.Linear, str],
    scales_dict: Dict[str, torch.Tensor],
    epsilon: float = 1e-6,
) -> Dict[str, torch.Tensor]:
    """
    Normalize per-layer scales across a group so that all resulting scaled weights
    have the same maximum dynamic range.

    This ensures that after applying the normalized scales:
        max(abs(weight / scale)) == shared_max (for all layers)

    Args:
        layers: List of nn.Linear layers in the group
        name_to_layer: Mapping from layer instance to its string name
        scales_dict: Dictionary of layer_name → best_scales (1D tensor per layer)
        epsilon: Small value to avoid division by zero

    Returns:
        Updated scales_dict with normalized per-channel scale tensors
    """
    per_layer_maxes = {}

    # Step 1: Compute max(abs(weight / scale)) for each layer
    for layer in layers:
        name = name_to_layer[layer]
        weight = layer.weight.detach()
        scale = scales_dict[name]

        # Ensure that scale is 1D of length out_features
        assert (
            weight.shape[0] == scale.shape[0]
        ), f"Mismatch: weight={weight.shape}, scale={scale.shape}"
        assert scale.dim() == 1
        assert scale.shape[0] == weight.shape[0]

        # Broadcast scale to match weight shape: [out, in]
        logger.debug(f"{name}: weight {weight.shape}, scale {scale.shape}")

        scaled_weight = weight / (scale.view(-1, 1) + epsilon)

        logger.debug(
            f"[normalize] {name} → weight: {weight.shape}, scale: {scale.shape}"
        )

        per_layer_maxes[name] = scaled_weight.abs().max().item()

    # Step 2: Find shared maximum across the group
    shared_max = max(per_layer_maxes.values())
    logger.debug(f"Shared max across group: {shared_max:.6f}")

    # Step 3: Normalize each layer's scales
    for layer in layers:
        name = name_to_layer[layer]
        original_scale = scales_dict[name]
        current_max = per_layer_maxes[name]

        ratio = current_max / (shared_max + epsilon)
        normalized_scale = original_scale * ratio

        logger.debug(
            f"[{name}] current_max={current_max:.6f}, "
            f"ratio={ratio:.6f}, "
            f"scale preview={normalized_scale[:5].tolist()}"
        )

        scales_dict[name] = normalized_scale

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
def apply_scale(
    module: nn.Module,
    scales_list: List[Tuple[str, Tuple[str], torch.Tensor]],
    input_feat_dict: Optional[Dict[str, torch.Tensor]] = None,
) -> None:
    """
    * Modified from standard code in scale.py

    Applies computed per-layer scales in-place to the model.

    Handles 3 connection patterns:
      1. Linear -> Linear
      2. Norm    -> Linear
      3. ActFn   -> Linear

    ---
    ASCII Diagram:
        ┌────────────┐     scale      ┌────────────┐
        │ prev_op    │ ────────────▶ │ Linear     │
        └────────────┘                └────────────┘

    Args:
        module (nn.Module): The top-level container.
        scales_list (List[Tuple]): Each tuple contains:
            - prev_op_name: str
            - (layer_name,): single-element tuple
            - scales: Tensor ∈ [out_features]
        input_feat_dict (Optional[Dict[str, Tensor]]): If provided, also scale
            input activations.

    Raises:
        NotImplementedError: If the pattern is unsupported.
        AssertionError: If layer isn't an nn.Linear or tuple length is not 1.
    """

    for prev_op_name, layer_names, scales in scales_list:
        assert len(layer_names) == 1, "Expected one target layer per scale tuple"
        layer_name = layer_names[0]

        prev_op = get_op_by_name(module, prev_op_name)
        layer = get_op_by_name(module, layer_name)

        assert isinstance(layer, nn.Linear), f"{layer_name} is not nn.Linear"

        best_device = get_best_device()
        prev_op.to(best_device)
        layer.to(best_device)
        scales = scales.to(best_device)

        logger.info(f"[apply_scale] Applying scale to {layer_name}")
        logger.debug(f"  - prev_op: {prev_op_name} ({type(prev_op).__name__})")
        logger.debug(f"  - scale.shape = {scales.shape}")
        logger.debug(f"  - weight.shape = {layer.weight.shape}")
        logger.debug(f"  - scale preview: {scales[:5].tolist()}")

        # Case 1: Linear -> Linear
        if isinstance(prev_op, nn.Linear):
            scale_fc_fc(prev_op, layer, scales)
            logger.debug(f"  - Pattern: Linear → Linear")

        # Case 2: Norm -> Linear
        elif (
            any(isinstance(prev_op, t) for t in allowed_norms)
            or "rmsnorm" in str(prev_op.__class__).lower()
        ):
            scale_ln_fc(layer, scales)
            logger.debug(f"  - Pattern: Norm → Linear")

        # Case 3: Activation -> Linear
        elif any(isinstance(prev_op, t) for t in allowed_act_fns):
            new_module = ScaledActivation(prev_op, scales)
            set_op_by_name(module, prev_op_name, new_module)
            scale_gelu_fc(prev_op, layer, scales)
            logger.debug(f"  - Pattern: Activation → Linear")

        else:
            raise NotImplementedError(
                f"Unsupported prev_op: {type(prev_op)}. "
                f"Allowed: Linear, {[t.__name__ for t in allowed_norms]}, "
                f"{[t.__name__ for t in allowed_act_fns]}"
            )

        if input_feat_dict and layer_name in input_feat_dict:
            inp = input_feat_dict[layer_name]
            inp.div_(scales.view(1, -1).to(inp.device))
            logger.debug(f"  - Scaled input_feat[{layer_name}] in-place")

        prev_op.cpu()
        layer.cpu()
        scales.cpu()
        logger.debug(f"  - Devices released (moved to CPU)")


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
