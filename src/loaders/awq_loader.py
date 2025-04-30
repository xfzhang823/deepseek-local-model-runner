"""
awq_loader.py

AWQ-only model loader — no bitsandbytes dependency.

This module defines a singleton-style model loader for AWQ-quantized
causal language models, supporting efficient inference with 4-bit weights
and optional safetensors format.

Environment Variables:
----------------------
MODEL_NAME_AWQ : str
    Absolute path to the directory containing the AWQ-quantized model.
    This directory must include:
        - model.safetensors or quantized weights
        - config.json with "quantization_config"
        - tokenizer.json, tokenizer_config.json
        - generation_config.json (optional)
        - special_tokens_map.json (optional)

Usage:
------
from loaders.awq_loader import AWQ_ModelLoader

tokenizer, model = AWQ_ModelLoader.load_model()

The model will be loaded into memory only once and reused on subsequent
calls.
"""

import os
import time
import logging
from typing import Tuple, Optional

from dotenv import load_dotenv
import transformers
from transformers import AutoTokenizer, AutoConfig
from awq import AutoAWQForCausalLM  # AWQ-specific quantized model class

logger = logging.getLogger(__name__)
load_dotenv()

# Load model path from .env
model_name = os.getenv("MODEL_NAME_AWQ")
if not model_name:
    raise EnvironmentError("❌ MODEL_NAME_AWQ not found in environment variables.")


class AWQ_ModelLoader:
    """
    AWQ_ModelLoader

    Singleton loader for AWQ-quantized language models.
    Loads the quantized model and tokenizer from the path specified by
    the environment variable MODEL_NAME_AWQ or an optional argument.

    This class avoids reloading the model on every request by caching
    the model and tokenizer as class-level attributes.

    Supported Features:
        - 4-bit quantized models (AWQ format)
        - SafeTensors format
        - trust_remote_code=True (for custom architectures like Qwen2)
        - Dynamic device mapping (GPU/CPU)
    """

    _model: Optional[AutoAWQForCausalLM] = None
    _tokenizer: Optional[AutoTokenizer] = None

    @classmethod
    def load_model(
        cls, model_path: str = model_name
    ) -> Tuple[AutoTokenizer, AutoAWQForCausalLM]:
        """
        Load the tokenizer and AWQ model from the given model directory path.

        This function ensures the model and tokenizer are loaded once and reused
        across calls. It prints the model type and transformers version for debugging.

        Parameters
        ----------
        model_path : str
            Path to the quantized model directory containing `config.json`,
            `model.safetensors`, and tokenizer files.

        Returns
        -------
        Tuple[AutoTokenizer, AutoAWQForCausalLM]
            - Tokenizer: Used to encode/decode text for the model.
            - Model: AWQ-quantized causal language model ready for inference.

        Raises
        ------
        Exception
            If loading the model or tokenizer fails.
        """
        if cls._model is None or cls._tokenizer is None:
            logger.info("🔄 Loading AWQ model from disk...")
            start = time.time()

            logger.info(f"📂 Loading AWQ quantized model")
            logger.info(f"🧠 Model path: {model_path}")
            logger.info(f"🔧 Transformers version: {transformers.__version__}")
            # # Optional: Log model_type from config
            # config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
            # logger.info(f"🧠 Detected model_type: {config.model_type}")
            # logger.debug(f"🔧 Transformers version: {transformers.__version__}")

            try:
                cls._model = AutoAWQForCausalLM.from_quantized(
                    quant_path=model_path,
                    trust_remote_code=True,
                    device_map=None,
                    fuse_layers=True,
                    safetensors=True,
                    use_exllama=False,
                    use_exllama_v2=False,
                )
                cls._tokenizer = AutoTokenizer.from_pretrained(
                    model_path, trust_remote_code=True
                )

                cls._model = cls._model.to("cuda")  # ✅ move after full loading

            except Exception as e:
                logger.exception("❌ Failed to load AWQ model.")
                raise e

            logger.info(f"✅ AWQ model loaded in {round(time.time() - start, 2)}s")
        else:
            logger.info("✅ Using cached AWQ model from memory")

        return cls._tokenizer, cls._model
