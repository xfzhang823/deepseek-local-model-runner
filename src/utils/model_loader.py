"""utils/model_loader.py"""

import logging
import os
from dotenv import load_dotenv
from typing import Tuple, Literal
from transformers import (
    BitsAndBytesConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)
import torch


logger = logging.getLogger(__name__)


def load_model(
    model_name: str, quantize: Literal["none", "8bit", "4bit"] = "none"
) -> Tuple[PreTrainedTokenizer | PreTrainedTokenizerFast, AutoModelForCausalLM]:
    """
    Load a model and tokenizer with optional quantization (8-bit or 4-bit).

    Args:
        model_name (str): The Hugging Face model identifier.
        quantize (Literal["none", "8bit", "4bit"]): Quantization mode.

    Returns:
        Tuple[AutoTokenizer, AutoModelForCausalLM]
    """
    load_dotenv()
    model_name = os.getenv("MODEL_NAME_HF", model_name)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if quantize == "8bit":
        logger.info(f"Loading model {model_name} with 8-bit quantization...")
        config = BitsAndBytesConfig(load_in_8bit=True, llm_int8_threshold=6.0)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            quantization_config=config,
        )

    elif quantize == "4bit":
        logger.info(f"Loading model {model_name} with 4-bit quantization...")
        config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            quantization_config=config,
        )

    else:
        logger.info(f"Loading model {model_name} without quantization...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
        )

    return tokenizer, model
