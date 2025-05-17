"""
scrooge_quant_pipeline.py

Full quantization pipeline for AWQ-style calibration.
"""

import os
import logging
import time
import torch
from dotenv import load_dotenv
from pathlib import Path
from typing import Optional

from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer, PreTrainedTokenizer
from quantize.scrooge_awq_quantizer import ScroogeAwqQuantizer, persist_awq_quantized_model
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
        device_map=None,  # Set this to None (manually manage)
        low_cpu_mem_usage=False,  # * Turn off meta device
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
    # Time log
    start_time = time.time()

    # Load full LLM model (base model)
    base_model = base_model or BASE_MODEL
    logger.info(f"Base model (full model): {base_model}")

    # Set up output dir
    save_dir_path = save_dir_path or OUT_DIR
    save_dir_path = str(Path(save_dir_path).expanduser().resolve())
    if not Path(save_dir_path).exists():
        raise FileNotFoundError(f"❌ Output dir does not exist: {save_dir_path}")

    logger.info(f"Output dir: {save_dir_path}")

    logger.info(f"🚀 Starting quantization for model: {base_model}")
    logger.info(f"Saving calibration to: {save_dir_path}")

    # Calbrate scales
    model, tokenizer = load_model_and_tokenizer(base_model)

    quantizer = ScroogeAwqQuantizer(
        model=model,
        tokenizer=tokenizer,
        max_calib_samples=max_calib_samples,
        max_calib_seq_len=max_calib_seq_len,
        apply_clip=apply_clip,
    )

    logger.info("🔧 Quantizer initialized.")

    # Step 1: Calibrate
    calib_stats_path = os.path.join(save_dir_path, "calib_stats.pt")

    # Check: if calibrate data file doesn't exist, calibrate
    if not os.path.exists(calib_stats_path):
        logger.info(f"📁 Calibration stats will be saved to: {calib_stats_path}")
        quantizer.calibrate(save_path=calib_stats_path, clear_inps=True)
        logger.info("✅ Calibration completed.")
    else:
        logger.info(
            f"📎 Found existing calibration file: {calib_stats_path} (skipping calibration)"
        )

    # Step 2: Quantization using saved stats (dict-based format)
    quantizer.modules = quantizer.get_model_layers(model)  # Ensure modules set

    logger.debug(f"📥 Loading calibration stats from: {calib_stats_path}")
    print(os.path.realpath(calib_stats_path))

    # Load calibration stats
    quantizer.load_calibration_stats(calib_stats_path)
    


    # Attach tokenizer to model so save_quantized() works
    model.tokenizer = tokenizer

    persist_awq_quantized_model(model, save_dir_path)
        logger.info("✅ Quantization pipeline completed successfully.")
    logger.info(
        f"✅ Quantization pipeline completed in {time.time() - start_time:.2f} seconds."
    )