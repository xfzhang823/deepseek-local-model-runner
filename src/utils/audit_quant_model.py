"""
utils/audit_quant_model.py
utils functions to inspect/audit safetensors file data.
"""

import logging
from typing import List
import torch.nn as nn
import torch
import pandas as pd
from awq.modules.linear import WQLinear_GEMM
from awq.utils.module import get_op_by_name
from transformers import PreTrainedModel


logger = logging.getLogger(__name__)


def audit_model_quantization(model: PreTrainedModel) -> pd.DataFrame:
    """
    Extended audit of model quantization state:
    - Classifies each layer as quantized or linear
    - Checks if weight/bias were loaded
    - Flags excluded layers (via config)
    - Detects presence of quantization artifacts (qweight, qzeros, scales)

    Args:
        model (PreTrainedModel): Loaded model object

    Returns:
        pd.DataFrame: Annotated table of quantization audit per layer

    Example:
        >>> from transformers import AutoAWQForCausalLM
        >>> model = AutoAWQForCausalLM.from_quantized("path/to/quant_model")
        >>> from quant_model_audit_utils import audit_model_quantization
        >>> audit_df = audit_model_quantization(model)
        >>> audit_df.to_csv("quant_audit_report.csv", index=False)
    """

    logger.info("Running quantization audit...")
    df = summarize_quantization_structure(model)

    try:
        state_keys = set(model.state_dict().keys())
    except Exception as e:
        logger.error("Could not access model state_dict: %s", e)
        raise

    exclude = set()
    try:
        if hasattr(model, "config"):
            exclude = set(model.config.quantization_config.get("exclude_layers", []))
    except Exception as e:
        logger.warning("Could not read quantization_config.exclude_layers: %s", e)

    def has_quant_artifacts(layer_name: str, model: torch.nn.Module) -> bool:
        """
        Checks whether the given layer has all expected quant artifacts:
        qweight, qzeros, scales.

        First tries state_dict, then falls back to runtime attributes.
        """
        state_dict = model.state_dict()
        base = layer_name.replace("model.model.", "model.", 1)

        keys_exist = all(
            f"{base}.{suffix}" in state_dict
            for suffix in ["qweight", "qzeros", "scales"]
        )

        if keys_exist:
            return True

        # Try direct access via runtime object
        try:
            layer = get_op_by_name(model, layer_name)
        except Exception:
            return False

        return all(
            hasattr(layer, attr) and isinstance(getattr(layer, attr), torch.Tensor)
            for attr in ["qweight", "qzeros", "scales"]
        )

    df["Excluded (Config)"] = df["Layer"].apply(
        lambda l: any(ex in l for ex in exclude)
    )
    df["Quant Artifacts"] = df["Layer"].apply(
        lambda layer_name: has_quant_artifacts(layer_name, model)
    )
    df["Quant State Expected"] = df["Module Type"] == "WQLinear_GEMM"

    logger.info("Quantization audit completed with %d layers checked.", len(df))
    return df


def audit_quantized_layers_in_memory(model):
    """
    Audits all quantized layers (WQLinear_GEMM) in the loaded model.
    Checks for presence and shape of qweight, qzeros, and scales tensors in memory.
    Logs a warning for any missing or zero-sized tensors.

    Example usage after model load:
    >>> audit_quantized_layers_in_memory(model)
    """
    missing_layers = []
    total_layers = 0

    for name, mod in model.named_modules():
        if isinstance(mod, WQLinear_GEMM):
            total_layers += 1
            qweight = getattr(mod, "qweight", None)
            qzeros = getattr(mod, "qzeros", None)
            scales = getattr(mod, "scales", None)
            has_all = True
            # Check presence and nonzero shape
            if (
                qweight is None
                or not isinstance(qweight, torch.Tensor)
                or qweight.numel() == 0
            ):
                logger.warning(f"[AUDIT] {name}: qweight MISSING or empty")
                has_all = False
            if (
                qzeros is None
                or not isinstance(qzeros, torch.Tensor)
                or qzeros.numel() == 0
            ):
                logger.warning(f"[AUDIT] {name}: qzeros MISSING or empty")
                has_all = False
            if (
                scales is None
                or not isinstance(scales, torch.Tensor)
                or scales.numel() == 0
            ):
                logger.warning(f"[AUDIT] {name}: scales MISSING or empty")
                has_all = False
            if has_all:
                qweight_shape = (
                    qweight.shape
                    if (qweight is not None and isinstance(qweight, torch.Tensor))
                    else None
                )
                qzeros_shape = (
                    qzeros.shape
                    if (qzeros is not None and isinstance(qzeros, torch.Tensor))
                    else None
                )
                scales_shape = (
                    scales.shape
                    if (scales is not None and isinstance(scales, torch.Tensor))
                    else None
                )
                logger.info(
                    f"[AUDIT] {name}: OK (qweight {qweight_shape}, qzeros {qzeros_shape}, scales {scales_shape})"
                )

            else:
                missing_layers.append(name)

    if missing_layers:
        logger.warning(
            f"⚠️ {len(missing_layers)} quantized layers with missing/invalid tensors:"
        )
        for lname in missing_layers:
            logger.warning(f"    - {lname}")
    else:
        logger.info("✅ All quantized layers loaded with valid tensors.")

    logger.info(f"Checked {total_layers} quantized layers.")


def summarize_quantization_structure(model: PreTrainedModel) -> pd.DataFrame:
    """
    Inspect and summarize quantization structure of a model:
    - Identifies WQLinear_GEMM (quantized) vs nn.Linear (unquantized)
    - Shows which tensors were loaded from state_dict (vs. missing or meta)

    Args:
        model (PreTrainedModel): A loaded model (e.g., from AutoAWQForCausalLM)

    Returns:
        pd.DataFrame: Summary with Layer, Module Type, Weight, Bias, Meta Tensor status

    Example:
        >>> from transformers import AutoAWQForCausalLM
        >>> model = AutoAWQForCausalLM.from_quantized("path/to/quant_model")
        >>> from quant_model_audit_utils import summarize_quantization_structure
        >>> df = summarize_quantization_structure(model)
        >>> print(df.head())
    """

    try:
        state_keys = set(model.state_dict().keys())
    except Exception as e:
        logger.error("Failed to retrieve model state_dict: %s", e)
        raise

    summary: List[dict] = []

    for name, module in model.named_modules():
        if isinstance(module, (WQLinear_GEMM, nn.Linear)):
            layer_type = (
                "WQLinear_GEMM" if isinstance(module, WQLinear_GEMM) else "nn.Linear"
            )
            w_key = f"{name}.weight"
            b_key = f"{name}.bias"

            weight_status = "loaded" if w_key in state_keys else "missing"
            bias_status = "loaded" if b_key in state_keys else "missing or not used"

            # Try checking meta status safely
            try:
                if hasattr(module, "weight"):
                    is_meta = getattr(module.weight, "is_meta", False)
                else:
                    is_meta = "n/a"
            except Exception as meta_error:
                logger.warning(
                    "Unable to determine 'meta' status for layer %s: %s",
                    name,
                    meta_error,
                )
                is_meta = "unknown"

            summary.append(
                {
                    "Layer": name,
                    "Module Type": layer_type,
                    "Weight": weight_status,
                    "Bias": bias_status,
                    "Meta Tensor": is_meta,
                }
            )

    df = pd.DataFrame(summary).sort_values("Layer").reset_index(drop=True)
    logger.info("Quantization structure summary completed with %d entries.", len(df))
    return df
