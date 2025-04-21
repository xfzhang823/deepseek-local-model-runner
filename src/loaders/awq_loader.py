"""
/loaders/awq_loader.py

AWQ-only model loader — no bitsandbytes dependency.
"""

import os
import time
import logging
from typing import Tuple, Optional

from dotenv import load_dotenv
from transformers import AutoTokenizer
from awq import AutoAWQForCausalLM  # AWQ-specific class

logger = logging.getLogger(__name__)


class AWQ_ModelLoader:
    """
    Singleton loader for AWQ-quantized models only.
    Assumes models are already quantized and locally cached.
    """

    _model: Optional[AutoAWQForCausalLM] = None
    _tokenizer: Optional[AutoTokenizer] = None

    @classmethod
    def load_model(cls) -> Tuple[AutoTokenizer, AutoAWQForCausalLM]:
        """
        Load and return the tokenizer and model using AWQ quantization.

        Returns:
            Tuple[AutoTokenizer, AutoAWQForCausalLM]: Tokenizer and model instances.
        """
        if cls._model is None or cls._tokenizer is None:
            logger.info("🔄 Loading AWQ model from disk...")
            start = time.time()

            load_dotenv()
            model_name = os.getenv("MODEL_NAME_AWQ")
            if not model_name:
                raise EnvironmentError(
                    "MODEL_NAME_AWQ not found in environment variables."
                )

            logger.info(f"Loading AWQ model: {model_name}")

            cls._model = AutoAWQForCausalLM.from_quantized(
                model_name,
                device_map="auto",
                trust_remote_code=True,
            )
            cls._tokenizer = AutoTokenizer.from_pretrained(model_name)

            logger.info(f"✅ AWQ model loaded in {round(time.time() - start, 2)}s")

        else:
            logger.info("✅ Using cached AWQ model from memory")

        return cls._tokenizer, cls._model
