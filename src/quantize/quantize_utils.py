"""quantize_utils.py"""

import os
import logging
from typing import Callable, Any, Dict, List, Tuple, Union, Optional
import torch
import torch.nn as nn
from torch import Tensor
from awq.modules.linear import WQLinear_GEMM
from awq.utils.module import set_op_by_name

# from utils.find_layer_type import is_attention_layer

logger = logging.getLogger(__name__)


class ScaleEntry:
    """
    A container class for storing named scaling factors with associated metadata.

    This class is designed to hold scaling parameters (typically as PyTorch tensors)
    along with their identifiers and optional sub-identifiers. Useful for managing
    normalization parameters, feature scaling factors, or other transformation parameters
    in machine learning pipelines.

    Attributes:
        name (str): Primary identifier for the scaling factor.
        value (torch.Tensor): The scaling parameter values stored as a PyTorch tensor.
        subnames (List[str]): Optional list of sub-identifiers, typically used when the
            scaling factor applies to multiple features or dimensions.

    Example:
        >>> # Creating a scale entry for image normalization
        >>> mean_scale = ScaleEntry(
        ...     name="image_normalization",
        ...     value=torch.tensor([0.485, 0.456, 0.406]),  # ImageNet mean
        ...     subnames=["red", "green", "blue"]
        ... )
        >>> print(mean_scale)
        ScaleEntry(name=image_normalization, subnames=['red', 'green', 'blue'], value_shape=torch.Size([3]))
    """

    def __init__(
        self, name: str, value: torch.Tensor, subnames: Optional[List[str]] = None
    ):
        """Initializes the ScaleEntry with name, value, and optional subnames.

        Args:
            name: Primary identifier for the scaling factor.
            value: The scaling parameter values as a PyTorch tensor.
            subnames: Optional list of sub-identifiers. If None, defaults to empty list.
        """
        self.name = name
        self.subnames = subnames or []
        self.value = value

    def __repr__(self):
        return f"ScaleEntry(name={self.name}, subnames={self.subnames}, value_shape={self.value.shape})"


def clear_up_module_memory(
    module: nn.Module,
    input_feat: Optional[Dict[str, torch.Tensor]] = None,
    device: Union[str, torch.device] = "cpu",
) -> None:
    """
    Move a module and its associated buffers to CPU, and clear memory-heavy dicts.

    This helps free VRAM during quantization workflows.

    Args:
        module (nn.Module): The model block to clean up.
        input_feat (Optional[Dict[str, Tensor]]): Input activations to clear.
        layer_kwargs (Optional[Dict[str, Tensor]]): Optional kwargs to move and clear.
        device (str | torch.device): Target device (usually 'cpu').
    """
    device = torch.device(device)
    logger.debug(f"🧹 Clearing module {module.__class__.__name__} to {device}")
    module.to(device)

    # Move rotary embedding cache if present
    if hasattr(module, "self_attn") and hasattr(module.self_attn, "rotary_emb"):
        re = module.self_attn.rotary_emb
        if getattr(re, "cos_cached", None) is not None:
            re.cos_cached = re.cos_cached.to(device)
            logger.debug("   ↳ Moved rotary_emb.cos_cached to CPU")
        if getattr(re, "sin_cached", None) is not None:
            re.sin_cached = re.sin_cached.to(device)
            logger.debug("   ↳ Moved rotary_emb.sin_cached to CPU")

    # Clear input features
    if input_feat:
        input_feat.clear()
        logger.debug("   ↳ Cleared input_feat")

    # Skip moving layer_kwargs — keep on GPU
    logger.debug("   ↳ Skipped moving layer_kwargs (kept on GPU)")

    # Trigger garbage collection and VRAM release
    torch.cuda.empty_cache()
    logger.debug("   ↳ Emptied CUDA cache")


def flatten_scales_or_clip_list(
    scales_or_clip_list: List[
        Union[ScaleEntry, Tuple[str, Union[Tuple[str], List[str]], torch.Tensor]]
    ],
) -> List[Tuple[str, torch.Tensor]]:
    """
    Flattens a list of scale tuples or ScaleEntry objects into a format compatible with `dict()`.

    Args:
        scales_or_clip_list: A list containing either ScaleEntry objects or tuples in
        the following formats:
            - 2-tuple: (layer_name: str, scale: Tensor)
            - 3-tuple: (prefix: str, subnames: Tuple[str] or List[str], scale: Tensor)

    Returns:
        A flat list of 2-tuples: List[(str, Tensor)], where each key is a dot-prefixed name
        like "prefix.subname".

    Raises:
        ValueError: If any entry is malformed or of an unexpected type.
    """
    flattened: List[Tuple[str, torch.Tensor]] = []

    for entry in scales_or_clip_list:

        # Handle ScaleEntry objects directly
        if isinstance(entry, ScaleEntry):
            for subname in entry.subnames:
                full_key = f"{entry.name}.{subname}"
                flattened.append((full_key, entry.value))
            if not entry.subnames:
                flattened.append((entry.name, entry.value))

        # Handle tuple structure
        elif isinstance(entry, tuple):
            if len(entry) == 2:
                key, tensor = entry
                if not isinstance(key, str) or not isinstance(tensor, torch.Tensor):
                    raise ValueError(f"Invalid 2-tuple entry: {entry}")
                flattened.append((key, tensor))

            elif len(entry) == 3:
                prefix, subkeys, tensor = entry
                if not isinstance(prefix, str) or not isinstance(tensor, torch.Tensor):
                    raise ValueError(f"Invalid 3-tuple entry: {entry}")
                if not isinstance(subkeys, (tuple, list)):
                    raise ValueError(f"Expected list or tuple for subkeys in: {entry}")

                for name in subkeys:
                    if not isinstance(name, str):
                        raise ValueError(f"Subkey must be a string in: {entry}")
                    full_key = f"{prefix}.{name}"
                    flattened.append((full_key, tensor))

            else:
                raise ValueError(f"Invalid entry length: {entry}")

        else:
            raise ValueError(f"Invalid entry type: {type(entry)} - {entry}")

    return flattened


def forward_with_memory_chunking(
    module: Callable[[Tensor], Tensor],
    inp: Tensor,
    module_kwargs: Dict[str, Any],
    max_chunk_memory: int,
) -> Tensor:
    """
    Run a forward pass on a module with large input tensor using memory-aware chunking.

    This function avoids GPU memory overflow by splitting the input into smaller chunks
    that fit within the specified `max_chunk_memory` limit. It processes each chunk
    sequentially, collects the outputs, and reconstructs the final output tensor.

    Args:
        module (Callable): A callable module (e.g., a linear layer) to apply to the input chunks.
        inp (Tensor): Input tensor of shape [B, S, D] where B is batch size, S is sequence length,
                      and D is input feature dimension.
        module_kwargs (Dict[str, Any]): Additional keyword arguments to pass into the module.
        max_chunk_memory (int): Maximum chunk memory in bytes (e.g., 64 * 1024 * 1024 for 64MB).

    Returns:
        Tensor: Output tensor of shape [B, S, ...], preserving batch and sequence dimensions.
    """
    # Flatten temporal dimension: [B, S, D] → [B*S, D]
    inp_flat = inp.view(-1, inp.shape[-1])

    # Estimate sizes
    num_rows = inp_flat.size(0)
    num_cols = inp_flat.size(1)
    element_size = inp_flat.element_size()  # in bytes (e.g., 2 for float16)

    # Calculate max rows per chunk to stay within memory budget
    chunk_size = max(1, max_chunk_memory // (element_size * num_cols))

    outputs: List[Tensor] = []
    for i in range(0, num_rows, chunk_size):
        chunk_inp = inp_flat[i : i + chunk_size].to(inp.device)
        out_chunk = module(chunk_inp, **module_kwargs)
        outputs.append(out_chunk.cpu())  # offload to CPU immediately
        torch.cuda.empty_cache()

    # Reconstruct final output: [B*S, ...] → [B, S, ...]
    return torch.cat(outputs, dim=0).to(inp.dtype).view(inp.shape[0], inp.shape[1], -1)


def get_scale_for_zero_point(
    weight: torch.Tensor, group_size: int, w_bit: int
) -> torch.Tensor:
    O, I = weight.shape
    G = I // group_size
    qmax = 2**w_bit - 1
    scales = torch.empty((G, O), dtype=torch.float32, device=weight.device)

    for g in range(G):
        start = g * group_size
        end = (g + 1) * group_size
        w_group = weight[:, start:end]
        w_min = w_group.min(dim=1).values
        w_max = w_group.max(dim=1).values
        scales[g] = (w_max - w_min).clamp(min=1e-5) / qmax

    return scales  # [G, O]


def get_safe_parallel_sample_count() -> int:
    """
    Get the SAFE no. of parallel sample, given the device's vram.
    If you flood the memory, the process will be VERY slow.
    """
    total_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    if total_mem < 6:
        return 4
    elif total_mem < 10:
        return 8
    else:
        return 16


def inspect_calib_stats(path: str):
    """
    Inspect and validate a saved calibration stats file.

    This will:
    - Load the file
    - Print type and structure of 'scales' and 'clips'
    - Preview several entries if they exist
    - Warn if anything looks inconsistent

    Example:
        >>> inspect_calib_stats("/path/to/calib_stats.pt")
    """
    print(f"\n📂 Inspecting: {path}", flush=True)

    data = torch.load(path, map_location="cpu")

    # Force dict casting for safety
    raw_scales = data.get("scales", {})
    raw_clips = data.get("clips", {})

    scales = dict(raw_scales or {})  # fallback to empty
    clips = dict(raw_clips or {})

    print(
        f"🔍 type(scales): {type(raw_scales)}, len: {len(scales)}, keys: {list(scales.keys())[:3]}",
        flush=True,
    )
    print(
        f"🔍 type(clips): {type(raw_clips)}, len: {len(clips)}, keys: {list(clips.keys())[:3]}",
        flush=True,
    )

    if len(scales) == 0:
        print(
            "⚠️ WARNING: 'scales' is empty — calibration may have failed or file is partial.",
            flush=True,
        )

    if len(scales) > 0:
        print("\n📏 Scales preview:")
        for k, v in list(scales.items())[:5]:
            print(
                f" - {k}: shape={tuple(v.shape)}, mean={v.mean():.4f}, std={v.std():.4f}"
            )
        if len(scales) > 5:
            print("   ...")

    if len(clips) > 0:
        print("\n✂️ Clips preview:")
        for k, v in list(clips.items())[:5]:
            print(
                f" - {k}: shape={tuple(v.shape)}, mean={v.mean():.4f}, std={v.std():.4f}"
            )
        if len(clips) > 5:
            print("   ...")

    print("")  # trailing newline for readability


def load_quantized_layers_into_model(
    model: nn.Module,
    load_dir: str,
    w_bit: int,
    group_size: int,
    in_features: int,
    out_features: int,
    device: str = "gpu",
) -> nn.Module:
    """
    Load quantized layers from disk and insert them into the model.

    Args:
        model (nn.Module): Model structure to populate.
        load_dir (str): Directory with .pt files.
        w_bit (int): Weight bit-width (usually 4).
        group_size (int): Quantization group size.
        in_features (int): Default in_features for WQLinear_GEMM.
        out_features (int): Default out_features for WQLinear_GEMM.
        device (str): Target device ("cpu" or "cuda").

    Returns:
        nn.Module: Model with layers replaced by WQLinear_GEMM.
    """
    for file_name in os.listdir(load_dir):
        if not file_name.endswith(".pt"):
            continue

        module_path = file_name[:-3]  # strip ".pt"
        file_path = os.path.join(load_dir, file_name)
        data = torch.load(file_path, map_location="cpu")

        bias = data.get("bias") is not None

        layer = WQLinear_GEMM(
            w_bit=w_bit,
            group_size=group_size,
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            dev=device,
        )

        layer.qweight = data["qweight"].to(device)
        layer.qzeros = data["qzeros"].to(device)
        layer.scales = data["scales"].to(device)

        if bias:
            layer.bias = data["bias"].to(device)

        set_op_by_name(model, module_path, layer)
        print(f"Loaded and set {module_path} from {file_name}")

    return model


def move_module_to_device(
    module: nn.Module,
    input_feat: Dict[str, torch.Tensor],
    scales: Optional[torch.Tensor] = None,
    zeros: Optional[torch.Tensor] = None,
    device: Union[str, torch.device] = "cuda",
) -> None:
    """
    Move a module and its related tensors to the specified device.

    Includes the module, input features, rotary cache, and optionally scales/zeros.

    Args:
        module (nn.Module): The model block to move.
        input_feat (Dict[str, torch.Tensor]): Activation inputs.
        scales (Optional[torch.Tensor]): Calibration scale tensor.
        zeros (Optional[torch.Tensor]): Calibration zero point tensor.
        device (str | torch.device): The target device.
    """
    device = torch.device(device)
    logger.debug(f"🔄 Moving module {module.__class__.__name__} to {device}")
    module.to(device)

    for name, tensor in input_feat.items():
        input_feat[name] = tensor.to(device)
        logger.debug(f"   ↳ Moved input_feat[{name}] to {device}")

    if scales is not None:
        scales.to(device)
        logger.debug("   ↳ Moved scales to device")
    if zeros is not None:
        zeros.to(device)
        logger.debug("   ↳ Moved zeros to device")

    if hasattr(module, "self_attn") and hasattr(module.self_attn, "rotary_emb"):
        re = module.self_attn.rotary_emb
        if getattr(re, "cos_cached", None) is not None:
            re.cos_cached = re.cos_cached.to(device)
            logger.debug("   ↳ Moved rotary_emb.cos_cached to device")
        if getattr(re, "sin_cached", None) is not None:
            re.sin_cached = re.sin_cached.to(device)
            logger.debug("   ↳ Moved rotary_emb.sin_cached to device")


def move_rope_to_device(
    model: torch.nn.Module, device: Union[str, torch.device]
) -> None:
    """
    Move rotary position embeddings (RoPE) to the specified device, if present.

    Args:
        model: The top-level model (may contain .model submodule).
        device: Target device (e.g., "cuda:0" or torch.device).
    """
    try:
        device = torch.device(device)  # ✅ Coerce to torch.device
        transformer = getattr(model, "model", model)
        if hasattr(transformer, "rotary_emb"):
            rope = transformer.rotary_emb
            if rope.device != device:
                transformer.rotary_emb = rope.to(device)
                logging.info(f"🔁 [RoPE] Moved rotary_emb to {device}")
    except Exception as e:
        logging.warning(f"⚠️ Failed to move rotary_emb to {device}: {e}")


def persist_quantized_layer(
    quant_layer: WQLinear_GEMM,
    save_dir: str,
    module_name: str,
    sub_layer_name: str,
) -> Optional[str]:
    """
    Save a single quantized WQLinear_GEMM layer to disk with a hierarchical name.

    Args:
        quant_layer (WQLinear_GEMM): The quantized linear layer.
        save_dir (str): Directory to save the quantized weights.
        module_name (str): Parent module name (e.g., "model.layers.5").
        sub_layer_name (str): Sub-layer name (e.g., "self_attn.q_proj").

    Returns:
        Optional[str]: Path to the saved file, or None if saving failed.

    Example:
    >>>    persist_quantized_layer(
    >>>        quant_layer=quantized_layer,
    >>>        save_dir=save_dir_path,
    >>>        module_name=f"model.layers.{idx}",
    >>>        sub_layer_name=layer_name
    >>>    )
    """
    try:
        if not isinstance(quant_layer, WQLinear_GEMM):
            raise TypeError(f"Expected WQLinear_GEMM, got {type(quant_layer)}")

        os.makedirs(save_dir, exist_ok=True)

        full_name = f"{module_name}.{sub_layer_name}"
        file_path = os.path.join(save_dir, f"{full_name}.pt")

        quant_data = {
            "qweight": quant_layer.qweight.clone().cpu(),
            "qzeros": quant_layer.qzeros.clone().cpu(),
            "scales": quant_layer.scales.clone().cpu(),
        }

        if quant_layer.bias is not None:
            quant_data["bias"] = quant_layer.bias.clone().cpu()

        torch.save(quant_data, file_path)
        logger.info(f"✅ Saved quantized layer: {file_path}")
        return file_path

    except Exception as e:
        logger.error(
            f"❌ Failed to save quantized layer '{module_name}.{sub_layer_name}': {e}",
            exc_info=True,
        )
        return None


def safe_update(
    target_list: List[Tuple[str, torch.Tensor]] | None,
    source: List[Tuple[str, torch.Tensor]],
    name: str = "scales",
    strict: bool = False,
) -> None:
    """
    Safely updates a target list with source list of (str, Tensor) pairs.

    Args:
        target_list (List[Tuple[str, torch.Tensor]]): The list to extend.
        source (List[Tuple[str, torch.Tensor]]): The source list of tuples.
        name (str, optional): Name for logging (e.g., "scales" or "clips").
            Defaults to "scales".
        strict (bool, optional): Whether to raise an error if structure is invalid.
            Defaults to False.

    Raises:
        TypeError: If strict=True and source is not a list of tuples.
        ValueError: If strict=True and source cannot be extended.
    """
    if not isinstance(source, list):
        message = (
            f"❌ [calibrate] {name}_list is not a list. Got {type(source)} instead."
        )
        if strict:
            logger.error(message)
            raise TypeError(f"{name}_list must be a list after processing.")
        else:
            logger.warning(message)
            return

    try:
        # Ensure the structure is List[Tuple[str, Tensor]]
        for entry in source:
            if not (
                isinstance(entry, tuple)
                and len(entry) == 2
                and isinstance(entry[0], str)
                and isinstance(entry[1], torch.Tensor)
            ):
                raise ValueError(f"Invalid entry in {name}_list: {entry}")

        # Extend the target list
        target_list.extend(source)
        logger.info(
            f"✅ [calibrate] {name.capitalize()} updated successfully with {len(source)} entries."
        )

    except Exception as e:
        message = f"❌ [calibrate] Failed to update {name} due to structure issue: {e}"
        if strict:
            logger.error(message)
            raise ValueError(
                f"{name}_list structure is invalid. Expected List[Tuple[str, Tensor]]."
            )
        else:
            logger.warning(message)
            logger.warning(
                f"🔍 [calibrate] Problematic {name}_list sample: {source[:5]}"
            )


def tensor_preview(t: torch.Tensor, rows: int = 3, cols: int = 3) -> str:
    try:
        if t.ndim == 1:
            return str(t[:rows].tolist())
        elif t.ndim == 2:
            return str(t[:rows, :cols].tolist())
        else:
            return f"Tensor of shape {tuple(t.shape)} — not previewed"
    except Exception as e:
        return f"[Tensor logging error: {e}]"


def unwrap_to_transformer(model: nn.Module) -> nn.Module:
    """
    Utils function to extract models from wrappers (common structure among LLMs).

    Traverse model wrappers to find the true transformer (e.g., Qwen2Model).

    Handles:
    - AutoAWQForCausalLM
    - Qwen2ForCausalLM
    - Qwen2Model
    """
    if hasattr(model, "model") and hasattr(model.model, "model"):
        # Example: AutoAWQForCausalLM -> Qwen2ForCausalLM -> Qwen2Model
        return model.model.model
    elif hasattr(model, "model") and hasattr(model.model, "layers"):
        # Example: Qwen2ForCausalLM -> Qwen2Model
        return model.model
    elif hasattr(model, "layers"):
        # Already unwrapped
        return model
    else:
        raise AttributeError(
            "Could not unwrap model to transformer. Unexpected structure."
        )
