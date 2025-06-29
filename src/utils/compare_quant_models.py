import torch
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json
import re
from pathlib import Path
import logging
from typing import Optional
from safetensors.torch import load_file


logger = logging.getLogger(__name__)


def load_tensor_from_safetensors(path: str | Path, tensor_key: str) -> torch.Tensor:
    """
    Load a single tensor from a .safetensors file.

    Args:
        path (str | Path): Path to the .safetensors file.
        tensor_key (str): Key name of the tensor to extract.

    Returns:
        torch.Tensor: Loaded tensor.
    """
    path = Path(path)
    tensor_dict = load_file(str(path))
    if tensor_key not in tensor_dict:
        raise KeyError(f"Tensor key '{tensor_key}' not found in {path.name}")
    return tensor_dict[tensor_key]


def load_tensor(path: str | Path, tensor_key: Optional[str] = None) -> torch.Tensor:
    """
    Load a tensor from either a .pt or .safetensors file.

    Args:
        path (Path | str): File path to load.
        tensor_key (Optional[str]): Required for safetensors; ignored for .pt.

    Returns:
        torch.Tensor
    """
    path = Path(path)
    suffix = path.suffix.lower()

    path = Path(path)
    if path.suffix == ".pt":
        obj = torch.load(path)
        if isinstance(obj, dict):
            if not tensor_key:
                raise ValueError(
                    f"tensor_key must be provided to extract from dict in {path}"
                )
            if tensor_key not in obj:
                raise KeyError(
                    f"Key '{tensor_key}' not found in dict loaded from {path}. Available keys: {list(obj.keys())}"
                )
            return obj[tensor_key]
        return obj
    elif suffix == ".safetensors":
        if not tensor_key:
            raise ValueError("tensor_key must be provided for safetensors")
        return load_tensor_from_safetensors(path, tensor_key)
    else:
        raise ValueError(f"Unsupported file format: {suffix}")


def sanitize(name: str) -> str:
    return re.sub(r"[^\w\-\.]", "_", name)


def save_diff_outputs(diff: torch.Tensor, base_name: str, save_dir: Path):
    npy_path = save_dir / f"{base_name}_diff.npy"
    json_path = save_dir / f"{base_name}_diff_preview.json"
    csv_path = save_dir / f"{base_name}_diff.csv"

    np.save(npy_path, diff.cpu().numpy())

    if diff.ndim == 2:
        pd.DataFrame(diff.cpu().numpy()).to_csv(csv_path, index=False)

    preview = diff[:10, :10].tolist() if diff.ndim == 2 else diff[:100].tolist()
    with open(json_path, "w") as f:
        json.dump(preview, f, indent=2)


def compare_qweights(
    model_1_tensor: torch.Tensor,
    model_2_tensor: torch.Tensor,
    label="qweight",
    save_dir: Optional[Path | str] = None,
) -> Optional[torch.Tensor]:
    if model_1_tensor.shape != model_2_tensor.shape:
        logger.warning("Shape mismatch in qweights.")
        return None

    diff = (model_1_tensor - model_2_tensor).float()
    logger.info(f"[qweight] Max diff: {diff.abs().max().item():.4f}")
    logger.info(f"[qweight] Mean diff: {diff.mean().item():.4f}")

    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        base_name = sanitize(label)
        save_diff_outputs(diff, base_name, save_dir)

        plt.figure(figsize=(10, 4))
        plt.plot(diff.flatten().cpu().numpy())
        plt.title(f"qweight Diff: {label}")
        plt.grid(True)
        plt.savefig(save_dir / f"{base_name}_diff_plot.png", bbox_inches="tight")
        plt.close()

    return diff


def compare_scales(
    scales_1: torch.Tensor,
    scales_2: torch.Tensor,
    label="scales",
    save_dir: Optional[Path | str] = None,
):
    if scales_1.shape != scales_2.shape:
        logger.warning("Shape mismatch in scales.")
        return None

    diff = (scales_1 - scales_2).abs()
    logger.info(f"[scales] Max diff: {diff.max().item():.6f}")
    logger.info(f"[scales] Mean diff: {diff.mean().item():.6f}")

    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        base_name = sanitize(label)
        save_diff_outputs(diff, base_name, save_dir)

        plt.figure(figsize=(10, 4))
        plt.plot(diff.cpu().numpy())
        plt.title(f"Scale Difference: {label}")
        plt.grid(True)
        plt.savefig(save_dir / f"{base_name}_diff_plot.png", bbox_inches="tight")
        plt.close()

    return diff


def compare_zero_points(
    zeros_1: torch.Tensor,
    zeros_2: torch.Tensor,
    label="qzeros",
    save_dir: Optional[Path | str] = None,
):
    if zeros_1.shape != zeros_2.shape:
        logger.warning("Shape mismatch in zero-points.")
        return None

    diff = (zeros_1 - zeros_2).abs().int()
    logger.info(f"[qzeros] Max diff: {diff.max().item()}")
    logger.info(f"[qzeros] Mean diff: {diff.float().mean().item():.4f}")

    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        base_name = sanitize(label)
        save_diff_outputs(diff, base_name, save_dir)

        plt.figure(figsize=(10, 4))
        plt.plot(diff.cpu().numpy())
        plt.title(f"Zero Point Difference: {label}")
        plt.grid(True)
        plt.savefig(save_dir / f"{base_name}_diff_plot.png", bbox_inches="tight")
        plt.close()

    return diff


def compare_tensors(
    model_1_file: Path | str,
    model_2_file: Path | str,
    tensor_key_prefix: str,
    save_dir: Path,
    suffixes=("qweight", "scales", "qzeros"),
    model_1_name: str = "model 1",
    model_2_name: str = "model 2",
):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    model_1_file = Path(model_1_file)
    model_2_file = Path(model_2_file)

    for suffix in suffixes:
        key = f"{tensor_key_prefix}.{suffix}"
        logger.info(f"🔎 Comparing: {key}")

        t1 = load_tensor(model_1_file, key)

        # Try to load using just the suffix; fallback to full key
        try:
            t2 = load_tensor(model_2_file, suffix)
        except KeyError:
            logger.warning(
                f"Suffix '{suffix}' not found in {model_2_file.name}. Trying full key '{key}'..."
            )
            t2 = load_tensor(model_2_file, key)

        if suffix == "qweight":
            compare_qweights(t1, t2, label=key, save_dir=save_dir)
        elif suffix == "scales":
            compare_scales(
                scales_1=t1,
                scales_2=t2,
                label=key,
                save_dir=save_dir,
            )
        elif suffix == "qzeros":
            compare_zero_points(
                zeros_1=t1,
                zeros_2=t2,
                label=key,
                save_dir=save_dir,
            )

        if suffix == "qweight":
            compare_qweights(t1, t2, label=key, save_dir=save_dir)
        elif suffix == "scales":
            compare_scales(
                scales_1=t1,
                scales_2=t2,
                label=key,
                save_dir=save_dir,
            )
        elif suffix == "qzeros":
            compare_zero_points(
                zeros_1=t1,
                zeros_2=t2,
                label=key,
                save_dir=save_dir,
            )
