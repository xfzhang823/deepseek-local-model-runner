"""
scrooge_quant_pipeline.py

Full quantization pipeline for AWQ-style calibration.
"""

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
from project_config import DEEPSEEK_R1_DISTILL_QUANT_MODEL_SCROOGE_DIR
from utils.vram_tracker import monitor_vram, VRAMMonitor
import logging_config  # assuming this sets up logging globally

logger = logging.getLogger(__name__)

load_dotenv()

# Load global config variables
BASE_MODEL = os.getenv(
    "MODEL_NAME_HF"
)  # Example: deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
HF_TOKEN = os.getenv("HUGGING_FACE_TOKEN")
OUT_DIR = str(DEEPSEEK_R1_DISTILL_QUANT_MODEL_SCROOGE_DIR / "quantized_layers")

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


def run_scrooge_quant_pipeline(
    base_model: Optional[str] = None,
    save_dir_path: Optional[str] = None,
    max_calib_samples: int = 96,
    max_calib_seq_len: int = 512,  # ! do not set this high
    apply_clip: bool = True,
    quant_config: Optional[Dict] = None,
    w_bit: int = 4,
    q_group_size: int = 128,
    zero_point: bool = True,
    version: str = "GEMM",
):
    """
    End-to-end Scrooge-style quantization pipeline.
    """
    start_time = time.time()

    base_model = base_model or BASE_MODEL
    if not base_model:
        raise ValueError(
            "❌ 'base_model' is not set. Please provide a model name or set the MODEL_NAME_HF environment variable."
        )

    # Normalize and resolve the save directory path
    raw_path = save_dir_path or OUT_DIR
    save_dir_path = str(Path(raw_path).expanduser().resolve(strict=False))
    logger.info(f"📁 Checking save directory: {save_dir_path}")

    # Proactively create or fail
    if os.path.exists(save_dir_path) and not os.path.isdir(save_dir_path):
        raise NotADirectoryError(
            f"❌ Path exists but is not a directory: {save_dir_path}"
        )

    os.makedirs(save_dir_path, exist_ok=True)
    logger.info(f"✅ Save directory is ready: {save_dir_path}")

    quant_config = quant_config or {
        "w_bit": w_bit,
        "q_group_size": q_group_size,
        "zero_point": zero_point,
        "version": version,
    }

    logger.info(f"🚀 Starting Scrooge quantization for: {base_model}")
    logger.info(f"Output directory: {save_dir_path}")

    log_resource_snapshot("BEFORE_MODEL_LOAD")
    model, tokenizer = load_model_and_tokenizer(base_model)  # type: ignore[arg-type]
    log_resource_snapshot("AFTER_MODEL_LOAD")

    quantizer = ScroogeAwqQuantizer(
        model=model,  # type: ignore[arg-type]
        tokenizer=tokenizer,
        quant_config=quant_config,
        max_calib_samples=max_calib_samples,
        max_calib_seq_len=max_calib_seq_len,
        apply_clip=apply_clip,
        save_dir=save_dir_path,
    )

    # ✅ Wrap with automatic VRAM monitor
    with monitor_vram(interval=2) as monitor:  # logging every 2 seconds
        logger.info("📊 Running calibration & quantization...")

        log_resource_snapshot("Before calibration & quantization")
        quantizer.calibrate_and_quantize(save_dir_path=save_dir_path)
        log_resource_snapshot("After calibration & quantization")

    # VRAM logging
    peak_vram = monitor_vram.get_peak_vram()
    logger.info(f"📈 Peak VRAM usage during quantization: {peak_vram:.2f} MB")

    logger.info("💾 Saving quantized model and config...")
    quantizer.save_quantized_and_configs(save_dir=save_dir_path)
    log_resource_snapshot("AFTER_SAVE")

    # Final timing
    elapsed = time.time() - start_time
    formatted_time = time.strftime("%H:%M:%S", time.gmtime(elapsed))
    logger.info(
        f"✅ Quantization pipeline completed in {formatted_time} ({elapsed:.2f} sec)"
    )
    resource_logger.info(f"[{base_model}] Quantization completed in {formatted_time}")
