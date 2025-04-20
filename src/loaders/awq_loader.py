import os
import time
import logging
from typing import Tuple, Optional
import awq
from awq import AutoAWQForCausalLM
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from dotenv import load_dotenv

   
logger = logging.getLogger(__name__)


class AWQ_ModelLoader:
    """
    Singleton loader for the model and tokenizer with support for both
    AWQ and BitsAndBytes quantization.
    """

    _model: Optional[AutoModelForCausalLM] = None
    _tokenizer: Optional[AutoTokenizer] = None
    _using_awq: bool = False

    @classmethod
    def load_model(cls) -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
        """
        Load and return the tokenizer and model, reusing them if already loaded.
        Automatically detects AWQ-quantized models.
        """
        if cls._model is None or cls._tokenizer is None:
            logger.info("🔄 Loading model from disk...")
            start = time.time()

            load_dotenv()
            model_name = os.getenv("MODEL_NAME")
            if not model_name:
                raise EnvironmentError(
                    "MODEL_NAME not found in .env or environment variables."
                )

            logger.info(f"Loading model: {model_name}")

            # Check if this is an AWQ-quantized model
            if AWQ_AVAILABLE and (
                "-awq" in model_name.lower() or "awq" in os.listdir(model_name)
            ):
                cls._using_awq = True
                logger.info("Detected AWQ-quantized model - using AWQ loader")

                # Load AWQ model
                cls._model = AutoAWQForCausalLM.from_quantized(
                    model_name, device_map="auto", trust_remote_code=True
                )
                cls._tokenizer = AutoTokenizer.from_pretrained(model_name)
            else:
                # Fallback to BitsAndBytes
                cls._using_awq = False
                logger.info("Using BitsAndBytes quantization")

                quant_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                )

                cls._tokenizer = AutoTokenizer.from_pretrained(model_name)
                cls._model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    device_map="auto",
                    quantization_config=quant_config,
                    trust_remote_code=True,
                )

            logger.info(f"Model loaded in {round(time.time() - start, 2)}s")
            logger.info(
                f"Quantization method: {'AWQ' if cls._using_awq else 'BitsAndBytes'}"
            )

        logger.info("✅ Using cached model from memory")
        return cls._tokenizer, cls._model

    @classmethod
    def generate_text(
        cls, prompt: str, max_new_tokens: int = 100, temperature: float = 0.7
    ) -> str:
        """
        Generate text using the loaded model
        """
        tokenizer, model = cls.load_model()

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
            )

        return tokenizer.decode(outputs[0], skip_special_tokens=True)
