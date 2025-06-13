"""
scrooge_quant_pipeline.py
"""

from pathlib import Path
import os
import logging
import torch
from dotenv import load_dotenv
from typing import Dict, Optional, Tuple
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast
from quantize.scrooge_awq_quantizer import ScroogeAwqQuantizer
from quantize.embed_batches_with_cache import embed_batches_with_cache
import logging_config
from project_config import DEEPSEEK_R1_DISTILL_QUANT_MODEL_OUTPUT_DIR


logger = logging.getLogger(__name__)

# Load full-precision model and tokenizer
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device for loading: {device}")


def load_project_config() -> Tuple[str, str, str, Path]:
    """
    Load base model, HF token, save directory, and embeddings file path.

    Returns:
        Tuple of:
            - base_model (str)
            - hf_token (str)
            - out_dir (str)
            - embeddings_file (Path)
    """
    load_dotenv()

    base_model = os.getenv("MODEL_NAME_HF")
    if base_model is None:
        raise ValueError("MODEL_NAME_HF must be set")

    hf_token = os.getenv("HUGGING_FACE_TOKEN")
    if hf_token is None:
        raise ValueError("HUGGING_FACE_TOKEN must be set")

    out_dir = str(DEEPSEEK_R1_DISTILL_QUANT_MODEL_OUTPUT_DIR)
    os.makedirs(out_dir, exist_ok=True)

    # This must always happen unconditionally before return
    embeddings_file = Path(
        "~/models/deepseek-awq-scrooge/calib_embeddings/full_embeddings.pt"
    ).expanduser()

    return base_model, hf_token, out_dir, embeddings_file


base_model, hf_token, out_dir, embeddings_file = load_project_config()


def load_full_model(base_model: str) -> Tuple[AutoAWQForCausalLM, PreTrainedTokenizer]:
    model = AutoAWQForCausalLM.from_pretrained(
        base_model,
        trust_remote_code=True,
        token=hf_token,
        device_map={"": device},  # Load to GPU if available
    )
    tokenizer = AutoTokenizer.from_pretrained(
        base_model,
        trust_remote_code=True,
        token=hf_token,
    )
    return model, tokenizer


def scrooge_quant_pipeline(
    base_model: Optional[str] = None,
    save_dir_path: Optional[str] = None,
    max_calib_samples: int = 512,
    max_calib_seq_len: int = 2048,
    apply_clip: bool = True,
    save_layers_dir: Optional[str] = None,
):
    """
    Full quantization pipeline:
    - load model,
    - calibrate with samples & save calibration,
    - apply quantization, & save quantized model.

    Args:
        - base_model (str): Path to the full model (FP16 or FP32) to quantize.
        - save_path (str): Path to save the final quantized model weights or statistics.
        - max_calib_samples (int, optional): Maximum number of calibration samples.
        Default is 512.
        - max_calib_seq_len (int, optional): Maximum sequence length for calibration
        samples.
        Default is 2048.
        - apply_clip (bool, optional): Whether to search for and apply clipping during
        calibration.
        Default is True.
        - save_layers_dir (str, optional): Directory to optionally save quantized
        individual layers immediately.

    Returns:
        quantizer (ScroogeAwqQuantizer): The quantizer object after quantization
        is completed.
    """

    # 1. Paths and model name
    base_model = base_model or os.getenv("MODEL_NAME_HF")
    if base_model is None:
        raise ValueError("Base model must be provided or set in environment.")

    save_dir_path = save_dir_path or str(DEEPSEEK_R1_DISTILL_QUANT_MODEL_OUTPUT_DIR)
    save_layers_dir = save_layers_dir or save_dir_path

    logger.info(f"Using base model: {base_model}")
    logger.info(f"Save directory: {save_dir_path}")
    logger.info(f"Save layers directory: {save_layers_dir}")

    # 2. Load the base model (FP16/FP32)
    model, tokenizer = load_full_model(base_model)

    # 3. Check for embedding files -> load or generate
    if not embeddings_file.exists():
        embeddings = embed_batches_with_cache(
            model=model,
            tokenizer=tokenizer,
            batch_size=16,
            max_seq_len=2048,
            embeds_path=embeddings_file,
        )
    else:
        embeddings = torch.load(embeddings_file)

    # 2. Initialize your Scrooge quantizer
    quantizer = ScroogeAwqQuantizer(
        model=model,
        tokenizer=tokenizer,
        max_calib_samples=max_calib_samples,
        max_calib_seq_len=max_calib_seq_len,
        apply_clip=apply_clip,
    )
    logger.info("quantizer initialized.")

    # todo: debug; delete later
    logger.info(
        f"🔍 Model type inside ScroogeQuantizer (quantizer.model): {quantizer.model.__class__}"
    )  # Debug — this should now be Qwen2Model

    # 3. Calibrate
    quantizer.calibrate_and_quantize(
        save_dir_path=save_dir_path,
        clear_inps=True,
    )

    logger.info(f"✅ Calibration finished. Stats saved at {save_dir_path}")

    # # 4. Apply quantization
    # quantizer.apply_quantization(
    #     calib_stats_path=save_path,  # Load the calibration file just saved
    #     save_layers_dir=save_layers_dir,  # Optional: Save quantized layers individually
    # )

    # # 5. Save final quantized model (optional step if not doing per-layer saving inside apply_quantization)
    # if save_layers_dir is None:
    #     # Save the entire model if layers weren't saved individually
    #     save_full_quantized_model(quantizer.model, output_dir=save_path)

    # return quantizer
