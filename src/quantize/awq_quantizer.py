"""
awq_quantizer.py

This module quantizes a full-precision Hugging-Face causal language model
(e.g., Qwen-2 1.5B) down to 4-bit AWQ format, and writes out a self-contained
directory that includes:

  ├── config.json              # original model config+quantization_config with data_type
  ├── generation_config.json   # sampling defaults
  ├── pytorch_model.safetensors  # 4-bit weight shards
  ├── tokenizer.json
  └── tokenizer_config.json

Environment variables (via .env or shell):
    MODEL_NAME_HF       The Hugging-Face repo for the full-precision base model
                        (e.g. "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B").
    HUGGING_FACE_TOKEN  (optional) HF access token for private models.
    AWQ_OUTPUT_DIR      (optional) Path to write the quantized model directory.
                        Defaults to "~/models/deepseek-awq".

Usage:
    python src/awq_quantizer.py
"""

import os
import logging
import torch
from dotenv import load_dotenv
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer
from datasets import load_dataset
import logging_config

logger = logging.getLogger(__name__)
load_dotenv()

# Verify GPU availability
if not torch.cuda.is_available():
    raise RuntimeError(
        "GPU is required but CUDA is not available. Please check CUDA setup."
    )

# Resolve paths and model name
base = os.getenv("MODEL_NAME_HF")  # e.g., "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
hf_token = os.getenv("HUGGING_FACE_TOKEN")
out_dir = os.path.expanduser(os.getenv("AWQ_OUTPUT_DIR", "~/models/deepseek-awq"))

# Ensure output directory is created
if not os.path.exists(out_dir):
    os.makedirs(out_dir, exist_ok=True)

logger.info(f"Base model: {base}")
logger.info(f"Output directory: {out_dir}")
logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")
logger.info(
    f"GPU memory available: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB"
)

# Clear GPU memory
torch.cuda.empty_cache()

# Load full-precision model and tokenizer
# device_map={"": "cuda"} loads directly to GPU, no model.to("cuda") needed
model = AutoAWQForCausalLM.from_pretrained(
    base, trust_remote_code=True, use_auth_token=hf_token, device_map={"": "cuda"}
)
tokenizer = AutoTokenizer.from_pretrained(
    base, trust_remote_code=True, use_auth_token=hf_token
)

# Debug: Check initial model state
logger.info("Initial model state:")
for name, param in model.named_parameters():
    if "weight" in name:
        logger.info(f"{name}: dtype={param.dtype}, device={param.device}")


# Load and process wikitext calibration data
def load_wikitext():
    try:
        logger.info("Loading wikitext dataset: wikitext-2-raw-v1")
        data = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        # Filter out empty or short samples, as per autoawq example
        calib_data = [
            text
            for text in data["text"]
            if text.strip() != "" and len(text.split(" ")) > 20
        ]
        logger.info(f"Loaded {len(calib_data)} calibration samples")
        return calib_data
    except Exception as e:
        logger.error(f"Failed to load wikitext dataset: {e}")
        raise


calib_data = load_wikitext()

# Quantize on GPU
quant_config = {"w_bit": 4, "q_group_size": 128, "zero_point": True, "version": "GEMM"}
try:
    model.quantize(
        tokenizer,
        quant_config=quant_config,
        export_compatible=True,
        calib_data=calib_data,
    )
except RuntimeError as e:
    logger.error(f"GPU quantization failed: {e}")
    raise

# Debug: Check quantization status
logger.info("Post-quantization model state:")
for name, param in model.named_parameters():
    if "weight" in name:
        logger.info(f"{name}: dtype={param.dtype}, device={param.device}")
meta_tensors = any(param.is_meta for param in model.parameters())
logger.info(f"Meta tensors after quantization: {meta_tensors}")

# Save quantized model and tokenizer
try:
    model.save_quantized(out_dir, safetensors=True)
    tokenizer.save_pretrained(out_dir)
    logger.info("✅ Saved quantized model and tokenizer successfully.")
    # Verify saved model size
    safetensors_file = os.path.join(out_dir, "model.safetensors")
    size_mb = os.path.getsize(safetensors_file) / (1024 * 1024)
    logger.info(f"Saved model.safetensors size: {size_mb:.2f} MB")
except Exception:
    logger.exception("❌ Failed to save quantized model.")
    raise
