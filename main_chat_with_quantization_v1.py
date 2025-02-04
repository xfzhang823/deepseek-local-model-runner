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
import logging
import re
from typing import Dict, List, Optional, Tuple
import torch
from dotenv import load_dotenv
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import logging_config

# Setup logging
logger = logging.getLogger(__name__)


def setup_device() -> torch.device:
    """
    Set up device (GPU if available, otherwise CPU).

    Returns:
        torch.device: The device to be used for computation.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    return device


def load_model(model_name: str) -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
    """
    Load the tokenizer and model.

    Args:
        model_name (str): The name or path of the pre-trained model.

    Returns:
        Tuple[AutoTokenizer, AutoModelForCausalLM]: The tokenizer and model.
    """
    logger.info(f"Loading tokenizer and model ({model_name})...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Load the model with 8-bit quantization
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",  # Automatically maps layers to GPU/CPU
            load_in_8bit=True,  # !Quantization part: enable 8-bit quantization
        )
        return tokenizer, model
    except Exception as e:
        logger.error(f"Failed to load model or tokenizer: {e}")
        raise


def create_pipeline(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    device: Optional[torch.device] = None,
    use_load_balancing: bool = False,
) -> object:
    """
    Create a pipeline for text generation.

    Args:
        - model (AutoModelForCausalLM): The pre-trained language model to use for text generation.
        - tokenizer (AutoTokenizer): The tokenizer to use for text preprocessing.
        *- device (Optional[torch.device], optional): The device to use for computation.
            Defaults to None.
        - use_load_balancing (bool, optional): Whether to use load balancing to dynamically allocate
            computation resources. If True, the device will not be set explicitly, allowing for
            dynamic allocation. If False, the device will be set based on the provided device argument.
            Defaults to False.

    Returns:
        object: The created pipeline object.

    Notes:
        *The option to set the device or not is provided to accommodate different use cases.
        When using dynamic load balancing (e.g., with Accelerate or set device_map to "auto"),
        which allows the load balancing mechanism to dynamically allocate resources,
        !do not set the device explicitly to avoid errors.
        However, when not using load balancing, set the device explicitly (to GPU usually).
    """
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
        logger.error(f"Failed to create pipeline: {e}")
        raise


def extract_thinking_and_response(text: str) -> Tuple[str, str]:
    """
    Extracts the "thinking" part and the actual response from the generated text.

    Args:
        text (str): The full generated text.

    Returns:
        Tuple[str, str]: A tuple containing the "thinking" part and the actual response.
    """
    logger.info("Extracting thinking and response parts...")

    # Check if response is None or empty
    if not text:
        logger.warning("Response is empty or None.")
        return "", ""

    # Extract the thinking part
    thinking_match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    if thinking_match:
        thinking_part = thinking_match.group(1).strip()
        logger.info("Thinking part found.")
    else:
        logger.warning("Thinking part not found.")
        thinking_part = ""

    # Extract the response part
    response_match = re.search(r"<response>(.*?)</response>", text, re.DOTALL)
    if response_match:
        response_part = response_match.group(1).strip()
        logger.info("Response part found.")
    else:
        logger.warning(
            "Response part not found. Assuming entire response is the final answer."
        )
        response_part = text.strip()

    return thinking_part, response_part


def generate_response(
    pipe: object,
    query: str,
    max_new_tokens: int = 8192,  # Set this high b/c R1 model has a "thinking" part
    truncation: bool = True,
    do_sample: bool = False,
    temperature: float = 0.7,
    max_retries: int = 3,  # Maximum number of retries for incomplete responses
) -> List[Dict[str, str]]:
    """
    Generate output using the pipeline.

    Args:
        - pipe (object): The text generation pipeline.
        - query (str): The input query for text generation.
        - max_new_tokens (int, optional): The maximum number of tokens to generate.
            Defaults to 8192.
        - truncation (bool, optional): Whether to truncate the input to fit within the model's
        max length.
            Defaults to True.
        - do_sample (bool, optional): Whether to use sampling for text generation.
            Defaults to False.
        - temperature (float, optional): The temperature for sampling. Defaults to 0.7.
        - max_retries (int, optional): Maximum number of retries for incomplete responses.
        Defaults to 3.

    Returns:
        List[Dict[str, str]]: A list of generated responses.
    """
    logger.info("Generating output...")
    for attempt in range(max_retries):
        try:
            messages = pipe(
                query,
                max_new_tokens=max_new_tokens,
                truncation=truncation,
                do_sample=do_sample,
                temperature=temperature,
            )

            logger.debug(f"Raw response: \n{messages}")  # todo: to debug; delete later

            full_response = "".join([message["generated_text"] for message in messages])

            logger.debug(f"Full Response: {full_response}")  # Debug logging

            # Check if the response is complete
            if "</response>" in full_response:
                logger.info("Full response generated.")
                return messages
            else:
                logger.warning(
                    f"Incomplete response detected (attempt {attempt + 1}/{max_retries}). Retrying..."
                )
        except Exception as e:
            logger.error(f"Failed to generate response: {e}")
            raise

    logger.error("Failed to generate a complete response after maximum retries.")
    raise RuntimeError("Failed to generate a complete response after maximum retries.")


def main() -> None:
    """
    Run the DeepSeek model testing pipeline.

    This function orchestrates the loading of the model, tokenizer, and pipeline,
    generates a response to a query, and logs/prints the results.
    """
    # Time tracking
    start_time = datetime.datetime.now()
    logger.info(f"Start time: {start_time.strftime('%Y-%m_%d %H:%M:%S')}")

    # Load environment variables
    load_dotenv()
    model_name = os.getenv("MODEL_NAME")
    if not model_name:
        logger.error("MODEL_NAME not found in environment variables.")
        raise ValueError("MODEL_NAME environment variable is required.")

    # Set up device
    device = setup_device()

    # Load model and tokenizer
    tokenizer, model = load_model(model_name)

    # Create pipeline
    pipe = create_pipeline(
        model=model,
        tokenizer=tokenizer,
        device=None,  # Set this to none b/c we are using Accelerate
        use_load_balancing=True,
    )

    # Base query
    # base_query = "Imagine an alien race is telepathic, will their ML system primarily reduce \
    # dimensionality or expand it?"

    base_query = "Where is the capital of China?"

    # todo: comment out for now to test
    # # Append the tagging instruction to the query
    # query = f"""
    # {base_query}

    # Please follow these instructions carefully:
    # 1. First, think through the problem step by step. Wrap your reasoning in <think> tags.
    # 2. Then, provide your final answer. Wrap your final answer in <response> tags.

    # <think>
    # [Your reasoning here]
    # </think>

    # <response>
    # [Your final answer here]
    # </response>
    # """

    query = base_query

    # Generate output
    messages = generate_response(pipe, query)

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
    logger.info(f"Elapsed Time: {elapsed_time}")


if __name__ == "__main__":
    main()
