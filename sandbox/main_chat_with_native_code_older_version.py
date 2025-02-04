""" With native code from DeepSeek"""

import logging
import os
from dotenv import load_dotenv
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import logging_config

# Setup logging
logger = logging.getLogger(__name__)

# Comment this out b/c using auto instead (dynamically switching between gpu and cpu)
# # Setup device parameter to move to GPU if available
# device = "cuda" if torch.cuda.is_available() else "cpu"

# * Step 1: Load the model and tokenizer
load_dotenv()
model_name = os.getenv("MODEL_NAME")
logger.info(f"Model name: {model_name}")


# Load model and quantize
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Define the quantization configuration for 8-bit
quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,  # Enable 8-bit quantization
    llm_int8_threshold=6.0,  # Threshold for 8-bit quantization
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",  #! dynamically balancing btw cpu and gpu
    quantization_config=quantization_config,  #! Quantization
)


# * Step 2: Define a function to generate responses
def generate_chat_response(
    prompt: str,
    max_length: int = 1000,
    temperature: float = 0.7,
    top_k: int = 50,
    top_p: float = 0.9,
):
    """
    This is a typical setup of temperature, top_k, and top_p.
    - Temperature (temperature):
    Temperature is a hyperparameter that controls the randomness or stochasticity of
    the generated text.
    It adjusts the distribution of probabilities over the possible next tokens in the sequence.
    A higher temperature value increases the randomness, resulting in more diverse and
    potentially more creative output, but also increases the risk of generating less coherent
    or less relevant text.
    Conversely, a lower temperature value produces more deterministic output, with less randomness
    and more predictable results.

    - Top-k (top_k):
    Top-k is a sampling strategy that limits the token choices to the top k most likely options,
    as determined by the model's probability distribution. By restricting the sampling space to
    the most probable tokens, top-k reduces the risk of generating irrelevant or low-quality text.
    However, it may also limit the model's ability to generate novel or creative content.

    - Top-p (top_p): Top-p, also known as nucleus sampling, is a sampling strategy that selects
    tokens dynamically until a cumulative probability threshold p is met.
    This approach allows the model to generate more diverse and coherent text by considering
    a wider range of possible tokens.
    Top-p is often used in combination with top-k to balance the trade-off between diversity
    and coherence.
    """
    # Add a "thinking" instruction to the prompt
    thinking_prompt = f"""
    Question: {prompt}
    <think>
    Let me think step by step:
    """

    # Tokenize the input prompt
    inputs = tokenizer(thinking_prompt, return_tensors="pt")

    # * Step 3: Generate intermediate outputs (thinking)
    print("Model is thinking...")
    with torch.no_grad():  # context manager disables gradient tracking for the block
        # Generate logits (intermediate outputs)
        logits = model(**inputs).logits
        print("Intermediate logits shape:", logits.shape)  # Debugging: inspect logits

        # Generate final response
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            do_sample=True,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )

    # *commented out because the R1 model includes reasoning in final output already.
    # intermediate_output = tokenizer.decode(
    #     logits[0].argmax(dim=-1), skip_special_tokens=True
    # )

    # * Step 4: Decode and display outputs
    # Decode the full response
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract the "thinking" part and the final answer
    if "<think>" in full_response and "</think>" in full_response:
        thinking_part = full_response.split("<think>")[1].split("</think>")[0].strip()
        final_answer = full_response.split("</think>")[1].strip()
    else:
        thinking_part = "No thinking steps captured."
        final_answer = full_response

    # Save the thinking part and final answer to variables
    thinking_output = thinking_part
    final_output = final_answer

    # Log the results
    logger.info(f"\nThinking Steps:\n{thinking_output}")
    logger.info(f"\nFinal Answer:\n{final_output}")

    # Return both the thinking part and final answer (if needed)
    return thinking_output, final_output


# Step 5: Run a simple chat loop
def main():
    """Orchestrate the loop"""
    print("Chat with DeepSeek R1! Type 'exit' to end the chat.")
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == "exit":
            break

        # Generate and display the response
        response = generate_chat_response(user_input)
        logger.info(response)
        print(f"DeepSeek: {response}")


if __name__ == "__main__":
    main()
