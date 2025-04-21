"""
/loaders/hf_loader.py

"""

import os
import time
import logging
from typing import Tuple
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from dotenv import load_dotenv


logger = logging.getLogger(__name__)


class HF_ModelLoader:
    """
    Singleton loader for the model and tokenizer.
    Ensures the model is only loaded once and reused across calls.
    """

    # class-level variable _models, and they will eventually hold a Hugging Face language
    # model, but are currently unset (None)
    _model: AutoModelForCausalLM = None
    _tokenizer: AutoTokenizer = None

    @classmethod
    def load_model(cls) -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
        """
        Load and return the tokenizer and model, reusing them if already loaded.

        Returns:
            Tuple[AutoTokenizer, AutoModelForCausalLM]: Tokenizer and model instances.
        """
        if (
            cls._model is None or cls._tokenizer is None
        ):  # * This keeps the model "warm"
            logger.info("🔄 Loading model from disk...")
            start = time.time()

            load_dotenv()
            model_name = os.getenv("MODEL_NAME_HF")

            if not model_name:
                raise EnvironmentError(
                    "MODEL_NAME not found in .env or environment variables."
                )

            logger.info(f"Loading model: {model_name}")
            tokenizer = AutoTokenizer.from_pretrained(model_name)

            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,  # * faster but slightly less accurate)
            )

            model = AutoModelForCausalLM.from_pretrained(
                model_name, device_map="auto", quantization_config=quant_config
            )  # device_map: dynamically balancing between CPU and GPU

            cls._model, cls._tokenizer = model, tokenizer

            logger.info("Model and tokenizer loaded and cached.")
            logger.info(f"Model loaded in {round(time.time() - start, 2)}s")

        logger.info("✅ Using cached model from memory")
        return cls._tokenizer, cls._model
