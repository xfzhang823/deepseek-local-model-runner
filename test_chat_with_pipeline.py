import logging
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Basic logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_model(model_name: str, test_prompt: str) -> None:
    """Simple test of model output"""
    try:
        # Load tokenizer
        logger.info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Load model with 8-bit quantization
        logger.info("Loading model...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name, device_map="auto", load_in_8bit=True
        )

        # Create pipeline
        logger.info("Creating pipeline...")
        device = 0 if torch.cuda.is_available() else -1
        pipe = pipeline(
            "text-generation", model=model, tokenizer=tokenizer, device=device
        )

        # Generate and log raw output
        logger.info("Generating response...")
        output = pipe(test_prompt, max_new_tokens=200, do_sample=True, temperature=0.7)

        logger.info("Raw output:")
        logger.info(output)

    except Exception as e:
        logger.error(f"Error occurred: {str(e)}")


if __name__ == "__main__":
    # Test parameters
    MODEL_NAME = (
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"  # Replace with your model name
    )
    TEST_PROMPT = "Where is the capital of China?"

    test_model(MODEL_NAME, TEST_PROMPT)
