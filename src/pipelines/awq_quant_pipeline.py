"""
awq_quant_pipeline.py

Refactored quantization pipeline for AWQ models using the standard AutoAWQ library.

This module encapsulates the AWQ quantization process into a reusable pipeline function
`run_awq_quant_pipeline` that can be easily invoked from main.py or other scripts.

The quantization process reduces a full-precision Hugging-Face causal language model
(e.g., Qwen-2 1.5B) down to 4-bit AWQ format and writes out a self-contained directory
that includes:

  ├── config.json            # Original model config and quantization settings
  ├── generation_config.json # Sampling defaults
  ├── model.safetensors      # 4-bit quantized weight shards
  ├── tokenizer.json         # Tokenizer configuration
  └── tokenizer_config.json  # Additional tokenizer settings

Environment variables (via .env or shell):
    MODEL_NAME_HF       The Hugging-Face repo for the full-precision base model
                        (e.g. "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B").
    HUGGING_FACE_TOKEN  (optional) HF access token for private models.
    AWQ_OUTPUT_DIR      (optional) Path to write the quantized model directory.
                        Defaults to "~/models/deepseek-awq".

Example usage:
    from awq_quant_pipeline import run_pipeline
    run_pipeline()
"""

import time
import os
import logging
import torch
from dotenv import load_dotenv
from awq import AutoAWQForCausalLM

# from awq.utils.calib_data import get_calib_dataset
from transformers import AutoTokenizer


logger = logging.getLogger(__name__)
load_dotenv()


def run_awq_quant_pipeline(
    n_samples: int = 64,  # defaul is usually 128
    max_seq_len: int = 512,
    n_parallel_calib_samples: int = 1,
    max_chunk_memory: int = 32 * 1024 * 1024,  # set to 64 MB (default is 1GB/very big)
    w_bit: int = 4,
    q_group_size: int = 128,
    zero_point: bool = True,
    version: str = "GEMM",
):
    """
    Execute the AWQ quantization process with the specified parameters.
    Includes timing and error handling for major steps.
    """
    torch.cuda.empty_cache()
    logger.info("Cleared GPU cache before model loading.")

    try:
        start_time = time.time()

        base = os.getenv("MODEL_NAME_HF")
        if base is None:
            raise ValueError("MODEL_NAME_HF must be set in the environment variables.")

        hf_token = os.getenv("HUGGING_FACE_TOKEN")
        out_dir = os.path.expanduser(
            os.getenv("AWQ_OUTPUT_DIR", "~/models/deepseek-awq")
        )
        os.makedirs(out_dir, exist_ok=True)

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA device is required for quantization.")

        torch.cuda.empty_cache()
        logger.info("Starting model loading...")
        model_load_start = time.time()
        model = AutoAWQForCausalLM.from_pretrained(
            base,
            trust_remote_code=True,
            use_auth_token=hf_token,
            device_map={"": "cuda"},
        )
        tokenizer = AutoTokenizer.from_pretrained(
            base, trust_remote_code=True, use_auth_token=hf_token
        )
        logger.info(f"Model loaded in {time.time() - model_load_start:.2f} seconds")

        logger.info("Loading calibration data...")
        calib_data_start = time.time()
        # raw_calib_data = get_calib_dataset(
        #     data="pileval",
        #     tokenizer=tokenizer,
        #     n_samples=n_samples,
        #     max_seq_len=max_seq_len,
        #     split="validation",
        #     text_column="text",
        # )
        logger.info(
            f"Calibration data loaded in {time.time() - calib_data_start:.2f} seconds"
        )

        quant_config = {
            "w_bit": w_bit,
            "q_group_size": q_group_size,
            "zero_point": zero_point,
            "version": version,
            # "n_parallel_calib_samples": n_parallel_calib_samples,
        }

        logger.info("Starting quantization...")
        quant_start = time.time()
        model.quantize(
            tokenizer=tokenizer,  # type: ignore
            quant_config=quant_config,
            export_compatible=True,
            calib_data="pileval",
            max_calib_samples=n_samples,
            max_calib_seq_len=max_seq_len,
            max_chunk_memory=max_chunk_memory,
            n_parallel_calib_samples=n_parallel_calib_samples,  # ! Lower to 2 if needed!
        )
        logger.info(
            f"Quantization completed in {time.time() - quant_start:.2f} seconds"
        )

        logger.info("Saving quantized model...")
        save_start = time.time()
        model.save_quantized(out_dir, safetensors=True)
        tokenizer.save_pretrained(out_dir)
        logger.info(
            f"Quantized model saved to {out_dir} in {time.time() - save_start:.2f} seconds"
        )

    except Exception as e:
        logger.error(f"Quantization pipeline failed: {e}")
        raise

    finally:
        elapsed = time.time() - start_time
        logger.info(
            f"Total AWQ quantization duration: {time.strftime('%H:%M:%S', time.gmtime(elapsed))}"
        )
