"""
scrooge_quant_pipeline.py

Full quantization pipeline for AWQ-style calibration.
"""

import os
import logging
import torch
from dotenv import load_dotenv
from pathlib import Path
from typing import Optional

from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer, PreTrainedTokenizer
from quantize.scrooge_awq_quantizer import ScroogeAwqQuantizer
from project_config import DEEPSEEK_R1_DISTILL_QUANT_MODEL_SCROOGE_DIR
import logging_config  # assuming this sets up logging globally

logger = logging.getLogger(__name__)
load_dotenv()

# Load global config variables
BASE_MODEL = os.getenv(
    "MODEL_NAME_HF"
)  # Example: deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
HF_TOKEN = os.getenv("HUGGING_FACE_TOKEN")
OUT_DIR = str(DEEPSEEK_R1_DISTILL_QUANT_MODEL_SCROOGE_DIR)

if BASE_MODEL is None:
    raise ValueError("Environment variable MODEL_NAME_HF must be set.")

os.makedirs(OUT_DIR, exist_ok=True)

logger.info(f"Model: {BASE_MODEL}")
logger.info(f"Output directory: {OUT_DIR}")


def load_model_and_tokenizer(
    base_model: str,
) -> tuple[AutoAWQForCausalLM, PreTrainedTokenizer]:
    """Load full precision model and tokenizer."""
    model = AutoAWQForCausalLM.from_pretrained(
        base_model,
        trust_remote_code=True,
        token=HF_TOKEN,
        device_map="auto",  # load efficiently
    )
    tokenizer = AutoTokenizer.from_pretrained(
        base_model,
        trust_remote_code=True,
        token=HF_TOKEN,
    )
    return model, tokenizer


def scrooge_quant_pipeline(
    base_model: Optional[str] = None,
    save_dir_path: Optional[str] = None,
    max_calib_samples: int = 96,
    max_calib_seq_len: int = 1024,
    apply_clip: bool = True,
):
    """
    Full Scrooge AWQ quantization pipeline:
    1. Load model and tokenizer
    2. Run calibration (live embeddings)
    3. Save calibration statistics

    Args:
        base_model (Optional[str]): Hugging Face model name or path.
        save_dir_path (Optional[str]): Where to save calibration statistics.
        max_calib_samples (int): Number of calibration samples to use.
        max_calib_seq_len (int): Max sequence length for calibration.
        apply_clip (bool): Whether to apply clipping search.
    """
    base_model = base_model or BASE_MODEL
    save_dir_path = save_dir_path or OUT_DIR

    logger.info(f"🚀 Starting quantization for model: {base_model}")
    logger.info(f"Saving calibration to: {save_dir_path}")

    model, tokenizer = load_model_and_tokenizer(base_model)

    quantizer = ScroogeAwqQuantizer(
        model=model,
        tokenizer=tokenizer,
        max_calib_samples=max_calib_samples,
        max_calib_seq_len=max_calib_seq_len,
        apply_clip=apply_clip,
    )

    logger.info("🔧 Quantizer initialized.")

    calib_stats_path = os.path.join(save_dir_path, "calib_stats.pt")
    quantizer.calibrate(
        save_path=calib_stats_path,
        clear_inps=True,
    )

    logger.info("✅ Quantization pipeline completed successfully.")
