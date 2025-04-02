"""
main_chat_wtih_native_code.py

This module loads a local causal language model, generates responses based on user input, 
and logs the thinking process along with the final answer. 

The model is loaded with quantization settings to optimize memory usage and performance 
(32 bits -> 8 bits).

The orchestration of the process is handled by the main function, and individual steps are 
broken down into dedicated functions.
"""

import logging
import os
from typing import Tuple
import time
from dotenv import load_dotenv
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import logging_config

# Setup logging
logger = logging.getLogger(__name__)


def load_model() -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
    """
    Load the model and tokenizer with 8-bit quantization configuration to optimize memory usage
    and inference performance.

    This function loads the model and tokenizer dynamically based on the environment variables.
    It also configures the model for 8-bit quantization, which reduces the memory footprint
    and speeds up inference, with the additional flexibility to control how much of the model
    is quantized via the 'llm_int8_threshold'.

    The quantization is performed using the `BitsAndBytesConfig`:
    - `load_in_8bit=True` ensures that the model weights are loaded in 8-bit precision (INT8),
        reducing the model's memory requirements.
    - `llm_int8_threshold=6.0` specifies a threshold for applying 8-bit quantization.
    Weights with magnitudes larger than this threshold will be quantized to 8-bit precision,
    while smaller weights may remain in higher precision to retain accuracy.

    * Why 6: More of a Rule of Thumb. The value 6.0 is a good starting point for
    * llm_int8_threshold based on empirical evidences, because it balances memory savings
    * and accuracy for many models.

    The model is loaded in a way that allows it to automatically balance between CPU
    and GPU resources.

    Returns:
        Tuple: The tokenizer and model objects.
    """
    load_dotenv()
    model_name = os.getenv("MODEL_NAME")
    logger.info(f"Loading model: {model_name}")

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Define the quantization configuration for 8-bit
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0,
    )

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",  #! Dynamically balancing between CPU and GPU
        quantization_config=quantization_config,  #! Quantization
    )

    logger.info(f"Model ({model_name}) loaded.")
    return tokenizer, model


def generate_chat_response(
    prompt: str,
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    max_length: int = 2000,
    temperature: float = 0.7,
    top_k: int = 50,
    top_p: float = 0.9,
) -> Tuple[str, str]:
    """
        Generate a response from the model based on the input prompt.

        Args:
            - prompt (str): The input prompt.
            - tokenizer (AutoTokenizer): The tokenizer to preprocess the input.
            - model (AutoModelForCausalLM): The model used for generating the response.
            - max_length (int): The maximum length of the generated output.
            - temperature (float): The randomness of the output.
            - top_k (int): The number of top token choices.
            - top_p (float): The cumulative probability threshold for nucleus sampling.

        Returns:
            Tuple[str, str]: The thinking steps and the final answer from the model.

    .   #* About temp, top_k, top_p
        Temperature controls the randomness of the generated text, with higher values
        leading to more creative but less coherent output, and lower values resulting
        in more predictable, deterministic responses.

        Top-k limits token choices to the top k most likely options, reducing irrelevant
        text but potentially limiting creativity.

        Top-p (nucleus sampling) selects tokens dynamically until a cumulative probability
        threshold is met, balancing diversity and coherence, often used in combination
        with top-k.
    """

    # Add a "thinking" instruction to the prompt
    thinking_prompt = f"""
    Question: {prompt}
    <think>
    Please reason through the problem step by step without repeating yourself. \
Each step should be concise and progress logically toward the final answer:
    """

    # Tokenize the input prompt
    inputs = tokenizer(thinking_prompt, return_tensors="pt")

    # Move input tensors to the same device as the model
    device = next(model.parameters()).device
    inputs = {key: value.to(device) for key, value in inputs.items()}

    # Start timing the thinking process
    start_time = time.time()

    # Generate logits and outputs
    with torch.no_grad():
        logits = model(**inputs).logits
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            do_sample=True,  # have multi-options (tokens) picks 1 based on prob.
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )

        logger.info(
            f"Intermediate logits shape: {logits.shape}"
        )  # Debugging: inspect logits

    elapsed_time = time.time() - start_time
    minutes, seconds = divmod(elapsed_time, 60)
    time_str = f"{int(minutes):02}:{int(seconds):02}"

    # Decode the full response
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract the "thinking" part and the final answer
    if "<think>" in full_response and "</think>" in full_response:
        thinking_part = full_response.split("<think>")[1].split("</think>")[0].strip()
        final_answer = full_response.split("</think>")[1].strip()
    else:
        thinking_part = "No thinking steps captured."
        final_answer = full_response

    # Log the thinking steps and final answer
    logger.info(f"Thinking time: {time_str}")
    logger.info(f"\nThinking Steps:\n{thinking_part}")
    logger.info(f"\nFinal Answer:\n{final_answer}")

    return thinking_part, final_answer


def main():
    """Orchestrate the loop"""
    print("Chat with DeepSeek R1! Type 'exit' to end the chat.")

    # Load the model and tokenizer
    tokenizer, model = load_model()

    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == "exit":
            break

        # Generate and display the response
        thinking_output, final_output = generate_chat_response(
            prompt=user_input, tokenizer=tokenizer, model=model, max_length=2000
        )
        print(f"DeepSeek (Thinking Steps): {thinking_output}")
        print(f"DeepSeek (Final Answer): {final_output}")
        logger.info(f"Response: {thinking_output} | {final_output}")


if __name__ == "__main__":
    main()
