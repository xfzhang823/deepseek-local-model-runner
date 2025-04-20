"""
DeepSeek Model Testing Module

This module provides a testing framework for the DeepSeek model, a quantized language model 
optimized for efficient inference.

The module utilizes the Hugging Face Transformers library to load the model and tokenizer,
and to generate text based on a given prompt.

The testing framework includes functionality for:
* Loading the DeepSeek model and tokenizer
* Creating a pipeline for text generation
* Generating text based on a given prompt
* Extracting thinking and response parts from the generated text
* Logging and printing the results

The module is designed to be used in a research or development setting, and provides 
a flexible framework for testing and evaluating the DeepSeek model.


*Quantization and Mixed Precision Training:
---------------------------------------------
- The DeepSeek model is quantized, which reduces memory usage and improves computational 
efficiency.
- Mixed precision training uses both 16-bit and 32-bit floating-point types to 
reduce memory usage and improve computational efficiency.
However, since this module is only used for inference (text generation), 
the GradScaler is not utilized.


Usage
-----
To use this module, simply import it and call the main function.
The main function will load the model and tokenizer, create a pipeline, generate text, 
and log and print the results.

Example
-------
from deepseek_model_testing import main
main()


Requirements
------------
* Python 3.8+
* Hugging Face Transformers library
* PyTorch library
* torch-quantization library (optional)

Note
----

This module is for research and development purposes only.
"""

# Dependencies
import os
import datetime
from datetime import timedelta
import logging
import re
from typing import Dict, List, Optional, Tuple, Any, Union
import torch
from dotenv import load_dotenv
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, Pipeline
from transformers.pipelines.text_generation import TextGenerationPipeline
import logging_config

# Setup logging
logger = logging.getLogger(__name__)


def setup_device() -> torch.device:
    """
    Set up and return the computation device (GPU if available, otherwise CPU).

    Returns:
        torch.device: The detected computation device.

    Raises:
        RuntimeError: If no valid computation device is found.
    """
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        return device
    except Exception as e:
        logger.error(f"Error setting up device: {str(e)}")
        raise RuntimeError(f"Failed to initialize computation device: {str(e)}")


def load_model(model_name: str) -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
    """
    Load the tokenizer and model from the specified model name.

    Args:
        model_name (str): Name or path of the model to load.

    Returns:
        Tuple[AutoTokenizer, AutoModelForCausalLM]: The loaded tokenizer and model.

    Raises:
        ValueError: If model_name is empty or invalid.
        RuntimeError: If model loading fails.
    """
    if not model_name:
        raise ValueError("Model name cannot be empty")

    logger.info("Loading tokenizer and model...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",  # Automatically maps layers to GPU/CPU
            load_in_8bit=True,  # Enable 8-bit quantization
        )
        return tokenizer, model
    except Exception as e:
        logger.error(f"Error loading model {model_name}: {str(e)}")
        raise RuntimeError(f"Failed to load model: {str(e)}")


def create_pipeline(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    device: Optional[torch.device] = None,
    use_load_balancing: bool = False,
) -> TextGenerationPipeline:
    """
    Create a pipeline for text generation.

    Args:
        model (AutoModelForCausalLM): The pre-trained language model to use for text generation.
        tokenizer (AutoTokenizer): The tokenizer to use for text preprocessing.
        device (Optional[torch.device]): The device to use for computation. Defaults to None.
        use_load_balancing (bool): Whether to use load balancing for dynamic resource allocation.
            If True, device will not be set explicitly. Defaults to False.

    Returns:
        TextGenerationPipeline: The created pipeline object.

    Raises:
        ValueError: If model or tokenizer is None.
        RuntimeError: If pipeline creation fails.

    Notes:
        The option to set the device or not is provided to accommodate different use cases.
        When using dynamic load balancing (e.g., with Accelerate), do not set the device explicitly
        will cause error, allowing the load balancing mechanism to dynamically allocate resources.
        However, when not using load balancing, set the device explicitly (to GPU usually).
    """
    if model is None or tokenizer is None:
        raise ValueError("Model and tokenizer must be provided")

    logger.info("Creating pipeline...")
    try:
        if use_load_balancing:
            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
            )
        else:
            device_id = 0 if device and device.type == "cuda" else -1
            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                device=device_id,
            )
        logger.info("Pipeline loaded.")
        return pipe
    except Exception as e:
        logger.error(f"Error creating pipeline: {str(e)}")
        raise RuntimeError(f"Failed to create pipeline: {str(e)}")


def extract_thinking_and_response(text: str) -> Tuple[str, str]:
    """
    Extracts the "thinking" part and the actual response from the generated text.

    Args:
        text (str): The full generated text.

    Returns:
        Tuple[str, str]: A tuple containing the "thinking" part and the actual response.
            Returns empty strings if no valid parts are found.

    Raises:
        ValueError: If input text is None or empty.
    """
    if not text:
        raise ValueError("Input text cannot be None or empty")

    logger.info("Extracting thinking and response parts...")

    thinking_part = ""
    response_part = ""

    try:
        # Check if thinking is in the text
        thinking_match = re.search(r"<think>.*?</think>", text, re.DOTALL)
        if thinking_match:
            thinking_part = (
                thinking_match.group(0)
                .replace("<think>", "")
                .replace("</think>", "")
                .strip()
            )
            logger.info("Thinking part found.")

            # Extract the response part
            response_part = re.sub(
                r"<think>.*?</think>", "", text, flags=re.DOTALL
            ).strip()
            if response_part:
                logger.info("Response part found.")
            else:
                logger.info("Response part not found.")
        else:
            logger.warning(
                "Thinking part not found. Assuming entire response is the final answer."
            )
            response_part = text.strip()

        return thinking_part, response_part
    except Exception as e:
        logger.error(f"Error extracting thinking and response: {str(e)}")
        return "", ""


def generate_response(
    pipe: TextGenerationPipeline,
    query: str,
    max_new_tokens: int = 400,
    truncation: bool = True,
    do_sample: bool = False,
    temperature: float = 0.7,
) -> List[Dict[str, Any]]:
    """
    Generate output using the text generation pipeline.

    Args:
        pipe (TextGenerationPipeline): The text generation pipeline.
        query (str): Input text prompt for generation.
        max_new_tokens (int): Maximum number of new tokens to generate. Defaults to 400.
        truncation (bool): Whether to truncate the input sequence. Defaults to True.
        do_sample (bool): Whether to use sampling for generation. Defaults to False.
        temperature (float): Sampling temperature. Defaults to 0.7.

    Returns:
        List[Dict[str, Any]]: List of generated responses.

    Raises:
        ValueError: If pipeline is None or query is empty.
        RuntimeError: If generation fails.
    """
    if pipe is None:
        raise ValueError("Pipeline cannot be None")
    if not query:
        raise ValueError("Query cannot be empty")

    logger.info("Generating output...")
    try:
        messages = pipe(
            query,
            max_new_tokens=max_new_tokens,
            truncation=truncation,
            do_sample=do_sample,
            temperature=temperature,
        )
        logger.info("Full response generated.")
        return messages
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        raise RuntimeError(f"Failed to generate response: {str(e)}")


def main() -> None:
    """
    Run the main testing workflow.

    This function orchestrates the entire testing process:
    - Sets up the environment and devices
    - Loads the model and creates the pipeline
    - Generates and processes responses
    - Handles timing and logging

    Raises:
        RuntimeError: If any critical step fails.
    """
    try:
        # Time tracking
        start_time = datetime.datetime.now()
        logger.info(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

        # Load environment variables
        load_dotenv()
        model_name = os.getenv("MODEL_NAME")
        if not model_name:
            raise ValueError("MODEL_NAME environment variable not set")

        # Set up device
        device = setup_device()

        # Load model and tokenizer
        tokenizer, model = load_model(model_name)

        # Create pipeline
        pipe = create_pipeline(model, tokenizer, device)

        # Base query
        base_query = "Imagine an alien race is telepathic, will their ML system primarily reduce \
dimensionality or expand it?"

        # Append the tagging instruction to the query
        query = f"""
        {base_query}

        Please provide your reasoning wrapped in <think> tags and your final response wrapped in <response> tags.
        """

        # Generate output
        messages = generate_response(pipe)

        # Collect the full response
        full_response = "".join([message["generated_text"] for message in messages])

        # Extract the thinking and response parts
        thinking, response = extract_thinking_and_response(full_response)

        # Log and print the results
        logger.info("Thinking Part:")
        logger.info(thinking)
        print("Thinking Part:")
        print(thinking + "\n")

        logger.info("Response Part:")
        logger.info(response)
        print("Response Part:")
        print(response + "\n")

        # Time tracking
        end_time = datetime.datetime.now()
        logger.info(f"End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        elapsed_time = end_time - start_time
        logger.info(f"Total execution time: {elapsed_time}")

    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise RuntimeError(f"Main execution failed: {str(e)}")


if __name__ == "__main__":
    main()
