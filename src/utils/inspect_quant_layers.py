"""
/utils/quant_layer_inspect.py

Utility module to inspect quantized layer files (.pt) for sanity checks.

Features:
- Validates presence and structure of `qweight`, `qzeros`, `scales`, and `bias` fields.
- Computes and flags high sparsity, outlier scale values, and NaN/Inf in tensors.
- Summarizes per-layer diagnostics and optionally saves to a timestamped log file.
- Designed for AWQ or Scrooge-style quantized LLM layer inspection.

Intended to be used during quantization validation or inference debugging.
"""

import logging
from pathlib import Path
from datetime import datetime
from typing import Optional
import torch


logger = logging.getLogger(__name__)


def inspect_qweight(qweight: torch.Tensor, zero_threshold: float = 0.9) -> str:
    """
    Inspect quantized weights for sparsity and unique values.

    Args:
        qweight (torch.Tensor): The quantized weight tensor.
        zero_threshold (float): Threshold to flag high sparsity.

    Returns:
        str: Summary string with shape, zero ratio, unique values, and flags.
    """
    zeros = (qweight == 0).sum().item()
    total = qweight.numel()
    zero_pct = zeros / total
    unique_vals = torch.unique(qweight).tolist()[:10]
    shape_str = str(tuple(qweight.shape))
    flag = "⚠️ HIGH ZERO RATIO!" if zero_pct > zero_threshold else "✅"
    return f"qweight: shape={shape_str}, zeros={zero_pct:.2%}, unique[:10]={unique_vals} {flag}"


def inspect_qzeros(qzeros: torch.Tensor) -> str:
    """
    Inspect zero-point tensor for sparsity.

    Args:
        qzeros (torch.Tensor): Zero-point tensor.

    Returns:
        str: Summary string with shape, sparsity ratio, and preview values.
    """
    nonzero = (qzeros != 0).sum().item()
    total = qzeros.numel()
    zero_pct = 1 - (nonzero / total)
    unique_vals = torch.unique(qzeros).tolist()[:10]
    shape_str = str(tuple(qzeros.shape))
    return (
        f"qzeros: shape={shape_str}, zero_pct={zero_pct:.2%}, unique[:10]={unique_vals}"
    )


def inspect_scales(scales: torch.Tensor) -> str:
    """
    Inspect scale tensor for distribution and numerical issues.

    Args:
        scales (torch.Tensor): Scale tensor.

    Returns:
        str: Summary with shape, stats, and NaN/Inf flags.
    """
    shape_str = str(tuple(scales.shape))
    stats = {
        "min": scales.min().item(),
        "max": scales.max().item(),
        "mean": scales.mean().item(),
    }
    has_nan = torch.isnan(scales).any().item()
    has_inf = torch.isinf(scales).any().item()
    outlier = stats["min"] < 1e-4 or stats["max"] > 10
    flag = "⚠️ NaN/Inf" if has_nan or has_inf else ("⚠️ Outliers" if outlier else "✅")
    return (
        f"scales: shape={shape_str}, min={stats['min']:.4g}, max={stats['max']:.4g}, "
        f"mean={stats['mean']:.4g} {flag}"
    )


def inspect_bias(bias: torch.Tensor) -> str:
    """
    Inspect bias tensor for distribution and NaNs.

    Args:
        bias (torch.Tensor): Bias tensor.

    Returns:
        str: Summary with shape and statistics.
    """
    shape_str = str(tuple(bias.shape))
    stats = {
        "min": bias.min().item(),
        "max": bias.max().item(),
        "mean": bias.mean().item(),
    }
    has_nan = torch.isnan(bias).any().item()
    flag = "⚠️ NaN" if has_nan else "✅"
    return f"bias: shape={shape_str}, min={stats['min']:.4g}, max={stats['max']:.4g}, mean={stats['mean']:.4g} {flag}"


def inspect_quant_layer_files(
    layer_dir: Path | str,
    zero_threshold: float = 0.9,
    save_path: Optional[Path | str] = None,
) -> None:
    """
    - Inspect all `.pt` quantized layer files in a directory for qweight, qzeros, scales,
    and bias sanity.
    - Save the inspection results to disk

    Args:
        layer_dir (Path): Directory containing quantized layer `.pt` files.
        zero_threshold (float): Sparsity threshold for `qweight` warnings.
        save_path (Optional[Path]): Optional file to write inspection results.

    Example:
        >>> from quant_layer_inspect import inspect_quant_layer_files
        >>> inspect_quant_layer_files(Path("~/models/deepseek-awq/quantized_layers"))
    """
    layer_dir = Path(layer_dir).expanduser()
    if not layer_dir.exists():
        raise FileNotFoundError(f"Directory not found: {layer_dir}")

    pt_files = sorted(
        layer_dir.glob("*.pt"),
        key=lambda f: [int(tok) if tok.isdigit() else tok for tok in f.stem.split(".")],
    )
    if not pt_files:
        print("❌ No .pt layer files found in directory.")
        return

    output_lines = []
    header = f"\n🔍 Inspecting {len(pt_files)} layer files in: {layer_dir}\n"
    logger.info(header)
    output_lines.append(header)

    valid_count = 0

    for f in pt_files:
        try:
            state_dict = torch.load(f, map_location="cpu")
            lines = [f"📂 {f.name}"]

            if "qweight" in state_dict:
                q_line = inspect_qweight(state_dict["qweight"], zero_threshold)
                lines.append(f"   ├─ {q_line}")
                if "✅" in q_line:
                    valid_count += 1
            else:
                lines.append("   ├─ qweight: ❌ missing")

            lines.append(
                f"   ├─ {inspect_qzeros(state_dict['qzeros'])}"
                if "qzeros" in state_dict
                else "   ├─ qzeros: ❌ missing"
            )
            lines.append(
                f"   ├─ {inspect_scales(state_dict['scales'])}"
                if "scales" in state_dict
                else "   ├─ scales: ❌ missing"
            )
            lines.append(
                f"   └─ {inspect_bias(state_dict['bias'])}"
                if "bias" in state_dict
                else "   └─ bias: (none)"
            )

            output_lines.extend(lines + [""])
            for line in lines:
                print(line)

        except Exception as e:
            err = f"❌ {f.name}: Error loading or parsing file — {str(e)}"
            output_lines.append(err)
            print(err)

    summary = f"\n📊 Summary: {valid_count}/{len(pt_files)} layers passed basic qweight check."
    logger.info(summary)
    output_lines.append(summary)

    if save_path:
        save_path = Path(save_path).expanduser()
        with open(save_path, "w", encoding="utf-8") as f:
            f.write("\n".join(output_lines))
        print(f"\n✅ Results saved to {save_path}")


if __name__ == "__main__":
    now = datetime.now()
    time_stamp = now.strftime("%Y-%m-%d_%H-%M")
    output_file_name = f"quant_layer_inspect_log_{time_stamp}.txt"

    quantized_dir = Path("~/models/deepseek-awq-scrooge/quantized_layers").expanduser()
    output_file = Path(
        "~/dev/deepseek_local_runner/documents", output_file_name
    ).expanduser()

    inspect_quant_layer_files(quantized_dir, save_path=output_file)
