"""
scrooge_quant_pipeline.py

Full quantization pipeline for AWQ-style calibration.
"""

from datetime import datetime
from pathlib import Path
import os
import logging
import time
from typing import Optional, Dict
from dotenv import load_dotenv
import psutil
from awq import AutoAWQForCausalLM
import torch
from transformers import AutoTokenizer, PreTrainedTokenizer

# From project modules
from quantize.scrooge_awq_quantizer import (
    ScroogeAwqQuantizer,
    # persist_awq_quantized_model,
)
from project_config import DEEPSEEK_R1_DISTILL_QUANT_MODEL_OUTPUT_DIR
from utils.vram_tracker import monitor_vram, VRAMMonitor
from utils.inspect_quant_layers import inspect_quant_layer_files
import logging_config  # assuming this sets up logging globally

logger = logging.getLogger(__name__)

load_dotenv()

# Load global config variables
BASE_MODEL = os.getenv(
    "MODEL_NAME_HF"
)  # Example: deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
HF_TOKEN = os.getenv("HUGGING_FACE_TOKEN")
OUT_DIR = str(DEEPSEEK_R1_DISTILL_QUANT_MODEL_OUTPUT_DIR)

if BASE_MODEL is None:
    raise ValueError("Environment variable MODEL_NAME_HF must be set.")

os.makedirs(OUT_DIR, exist_ok=True)

logger.info(f"Model: {BASE_MODEL}")
logger.info(f"Output directory: {OUT_DIR}")


resource_logger = logging.getLogger("resource")


def log_resource_event(message: str):
    """Write a custom message to the resource log."""
    resource_logger.info(message)


def log_resource_snapshot(tag: str = "RESOURCE") -> None:
    """
    Log CPU, RAM, and GPU memory usage under the 'resource' logger.
    """

    resource_logger = logging.getLogger("resource")

    cpu = psutil.cpu_percent()
    ram_gb = psutil.virtual_memory().used / (1024**3)
    msg = f"[{tag}] CPU: {cpu:.1f}% | RAM: {ram_gb:.2f} GB"

    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        vram_mb = torch.cuda.memory_allocated(device) / (1024**2)
        vram_total = torch.cuda.get_device_properties(device).total_memory / (1024**2)
        msg += f" | GPU {device}: VRAM {vram_mb:.1f}/{vram_total:.1f} MB"

    resource_logger.info(msg)


def load_model_and_tokenizer(
    base_model: str,
) -> tuple[AutoAWQForCausalLM, PreTrainedTokenizer]:
    """Load full precision model and tokenizer."""
    model = AutoAWQForCausalLM.from_pretrained(
        base_model,
        trust_remote_code=True,
        token=HF_TOKEN,
        device_map=None,  # Set this to None (manually manage)
        low_cpu_mem_usage=False,  # * Turn off meta device
    )
    tokenizer = AutoTokenizer.from_pretrained(
        base_model,
        trust_remote_code=True,
        token=HF_TOKEN,
    )
    return model, tokenizer


def prepare_output_dir(path: Optional[str], fallback: str, label: str) -> str:
    """
    Resolve, validate, and create output directory if needed.

    Args:
        path (Optional[str]): User-supplied path.
        fallback (str): Default path if none is provided.
        label (str): Label for logging.

    Returns:
        str: Absolute directory path.
    """
    resolved_path = str(Path(path or fallback).expanduser().resolve(strict=False))
    logger.info(f"📁 Checking {label} directory: {resolved_path}")

    if os.path.exists(resolved_path) and not os.path.isdir(resolved_path):
        raise NotADirectoryError(
            f"❌ Path exists but is not a directory: {resolved_path}"
        )

    os.makedirs(resolved_path, exist_ok=True)
    logger.info(f"✅ {label} directory is ready: {resolved_path}")
    return resolved_path


def run_scrooge_quant_pipeline(
    base_model: Optional[str] = None,
    save_layers_dir: Optional[str] = None,
    save_model_dir: Optional[str] = None,
    max_calib_samples: int = 96,
    max_calib_seq_len: int = 512,
    apply_clip: bool = True,
    quant_config: Optional[Dict] = None,
    w_bit: int = 4,
    q_group_size: int = 128,
    zero_point: bool = True,
    version: str = "GEMM",
):
    """
    Runs the full Scrooge-style quantization pipeline for a given HuggingFace model.

    This includes model loading, calibration, quantization, saving quantized weights,
    and logging system resources (e.g., VRAM usage) during the process.

    Args:
        base_model (str, optional): HuggingFace model identifier or local path. If None,
            falls back to the `MODEL_NAME_HF` environment variable.
        save_layers_dir (str, optional): Directory for storing per-layer quantized weights.
            Defaults to `quantized_layers` inside `OUT_DIR`.
        save_model_dir (str, optional): Directory for saving the fully integrated quantized model.
            Defaults to `quantized_model` inside `OUT_DIR`.
        max_calib_samples (int): Number of calibration samples to use.
        max_calib_seq_len (int): Maximum sequence length for calibration.
        apply_clip (bool): Whether to apply clipping during quantization.
        quant_config (dict, optional): Custom quantization config; overrides w_bit,
            q_group_size, etc.
        w_bit (int): Weight bit-width (e.g., 4).
        q_group_size (int): Group size for weight quantization.
        zero_point (bool): Whether to apply asymmetric quantization.
        version (str): Backend implementation (e.g., "GEMM", "GEMV").

    Raises:
        ValueError: If `base_model` is not provided and no environment fallback is found.
        NotADirectoryError: If any save path exists and is not a directory.
    """
    start_time = time.time()

    base_model = base_model or BASE_MODEL
    if not base_model:
        raise ValueError(
            "❌ 'base_model' is not set. Please provide a model name or set the MODEL_NAME_HF environment variable."
        )

    # Default directories
    if save_layers_dir is None:
        save_layers_dir = save_layers_dir or os.path.join(OUT_DIR, "quantized_layers")
    if save_model_dir is None:
        save_model_dir = save_model_dir or os.path.join(OUT_DIR, "quantized_model")

    if save_layers_dir == save_model_dir:
        raise ValueError(
            "❌ save_layers_dir and save_model_dir must be different directories. "
            f"Both are set to: {save_layers_dir}"
        )

    save_layers_dir = prepare_output_dir(save_layers_dir, OUT_DIR, "save_layers")
    save_model_dir = prepare_output_dir(save_model_dir, OUT_DIR, "save_model")

    quant_config = quant_config or {
        "w_bit": w_bit,
        "q_group_size": q_group_size,
        "zero_point": zero_point,
        "version": version,
        "quant_method": "awq",
    }

    logger.info(f"🚀 Starting Scrooge quantization for: {base_model}")
    logger.info(f"Quantization config: {quant_config}")
    logger.info(f"Per-layer output directory: {save_layers_dir}")
    logger.info(f"Final model output directory: {save_model_dir}")

    log_resource_snapshot("BEFORE_MODEL_LOAD")
    model, tokenizer = load_model_and_tokenizer(base_model)
    log_resource_snapshot("AFTER_MODEL_LOAD")

    quantizer = ScroogeAwqQuantizer(
        model=model,
        tokenizer=tokenizer,
        quant_config=quant_config,
        max_calib_samples=max_calib_samples,
        max_calib_seq_len=max_calib_seq_len,
        apply_clip=apply_clip,
    )

    # * ✅ Quantize and persist layers
    with monitor_vram(interval=2) as monitor:
        logger.info("📊 Running calibration and quantization...")

        log_resource_snapshot("BEFORE_CALIBRATE")
        quantizer.calibrate_and_quantize(
            save_dir_path=save_layers_dir,
            use_full_calib=False,  # ! Set to use simple calibration
        )
        log_resource_snapshot("AFTER_CALIBRATE")

        peak_vram = monitor.get_peak_vram()

    logger.info(f"📈 Peak VRAM usage during quantization: {peak_vram:.2f} MB")

    # * ✅ Inspect quantized layer files
    now = datetime.now()
    time_stamp = now.strftime("%Y-%m-%d_%H-%M")
    output_file_name = f"quant_layer_inspect_log_{time_stamp}.txt"

    output_file = Path(
        "~/dev/deepseek_local_runner/documents", output_file_name
    ).expanduser()
    inspect_quant_layer_files(save_layers_dir, save_path=output_file)
    logger.info(f"Quantized layers inspection saved to {output_file}.")

    # * ✅ Load layers, integrate/build model, persist model files to disk
    logger.info("💾 Loading layers and saving quantized model and configuration...")

    # * Re-initiate the model again
    # # ✅ Reload full model to apply layer patches cleanly
    # model, tokenizer = load_model_and_tokenizer(base_model)
    # quantizer = ScroogeAwqQuantizer(
    #     model=model,
    #     tokenizer=tokenizer,
    #     quant_config=quant_config,
    #     max_calib_samples=max_calib_samples,
    #     max_calib_seq_len=max_calib_seq_len,
    #     apply_clip=apply_clip,
    # )
    quantizer.build_and_persist_model(load_dir=save_layers_dir, save_dir=save_model_dir)

    saved_files = [f.name for f in Path(save_model_dir).iterdir() if f.is_file()]
    logger.info(f"Model files saved in {save_model_dir}: {saved_files}")

    log_resource_snapshot("AFTER_SAVE")

    elapsed = time.time() - start_time
    formatted_time = time.strftime("%H:%M:%S", time.gmtime(elapsed))

    logger.info(
        f"✅ Quantization pipeline completed in {formatted_time} ({elapsed:.2f} sec)"
    )
    resource_logger.info(f"[{base_model}] Quantization completed in {formatted_time}")
