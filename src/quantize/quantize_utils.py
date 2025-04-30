"""quantize_utils.py"""

from typing import Any, Dict
import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)


def safe_update(
    target_dict: Dict[str, torch.Tensor],
    source: Any,
    name: str = "scales",
    strict: bool = False,
) -> None:
    """
    Safely updates a target dictionary from a source list of (str, Tensor) pairs.

    Args:
        - target_dict (Dict[str, torch.Tensor]): The dictionary to update.
        - source (Any): The object expected to be a list of (str, Tensor) tuples.
        - name (str, optional): Name for logging (e.g., "scales" or "clips").
        Defaults to "scales".
        - strict (bool, optional): Whether to raise an error if structure is invalid.
        Defaults to False.

    Raises:
        TypeError: If strict=True and source is not a list.
        ValueError: If strict=True and source cannot be converted to a dict.
    """
    if not isinstance(source, list):
        message = (
            f"❌ [calibrate] {name}_list is not a list. Got {type(source)} instead."
        )
        if strict:
            logger.error(message)
            raise TypeError(f"{name}_list must be a list after append_str_prefix().")
        else:
            logger.warning(message)
            return  # Quietly skip update if not strict

    try:
        source_dict = dict(source)
        target_dict.update(source_dict)
        logger.info(
            f"✅ [calibrate] {name.capitalize()} updated successfully with {len(source_dict)} entries."
        )
    except Exception as e:
        message = f"❌ [calibrate] Failed to update {name} due to structure issue: {e}"
        if strict:
            logger.error(message)
            logger.error(f"🔍 [calibrate] Problematic {name}_list sample: {source[:5]}")
            raise ValueError(
                f"{name}_list structure is invalid. Expected List[Tuple[str, Tensor]]."
            )
        else:
            logger.warning(message)
            logger.warning(
                f"🔍 [calibrate] Problematic {name}_list sample: {source[:5]}"
            )
            return  # Quietly skip update if not strict


def unwrap_to_transformer(model: nn.Module) -> nn.Module:
    """
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
