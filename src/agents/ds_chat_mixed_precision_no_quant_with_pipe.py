"""
Testing Chat with DeepSeek model.

!This version included mixed precision; however, it still took over 40 min and it was cut off, 
!despite setting max_tokens at very high.

Mixed precision training uses both 16-bit and 32-bit floating-point types to 
reduce memory usage and improve computational efficiency.

The GradScaler is typically used during training when you have a loss that you backpropagate. 
In our case, since we are only doing inference (text generation), you might not need to use it.
"""

import textwrap
import os
import datetime
import logging
from dotenv import load_dotenv
from transformers import pipeline
import torch
from torch.amp import autocast
import logging_config

# Constants
QUERY = "Imagine an alien race is telepathic, will their ML system primarily reducing dimensionality \
or expanding it?"

# Setup logging
logger = logging.getLogger(__name__)

# Time tracking
start_time = datetime.datetime.now()
logger.info(f"Start time: {start_time.strftime('%Y-%m_%d %H:%M:%S')}")


# Set up device (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")


def print_response(text_data) -> None:
    # Extract the text
    text = next(
        (item["generated_text"] for item in text_data if "generated_text" in item), None
    )

    parsed_text = text.replace("\n\n", "\n")
    paragraphs = parsed_text.split("\n")

    for paragraph in paragraphs:
        if paragraph:  # Check if paragraph is not empty
            formatted_paragraph = textwrap.fill(paragraph, width=100)
            print(formatted_paragraph + "\n")  # Add extra line spacing


# Model name from .env
load_dotenv()
model = os.getenv("MODEL_NAME")

logger.info("Loading pipeline...")
pipe = pipeline("text-generation", model=model)
logger.info("Pipeline loaded.")


try:
    # Use autocast for mixed precision (only on GPU)
    if device.type == "cuda":
        with autocast(
            dtype=torch.float16, device_type="cuda"
        ):  # Enable mixed precision on GPU
            logger.info("Generating output...")
            messages = pipe(QUERY, max_length=7000, truncation=True)
    else:
        logger.info("Generating output...")
        messages = pipe(QUERY, max_new_length=4000, truncation=True)

    logger.info("Printing Response...")
    print_response(messages)

except Exception as e:
    logger.error(f"Error occurred: {e}")


# Time tracking
end_time = datetime.datetime.now()
logger.info(f"End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")

elapsed_time = end_time - start_time
logger.info(f"Elapsed Time: {elapsed_time}")


# query = "There are three switches, but they are not labelled. Each switch corresponds to one of three light bulbs in a room. \
# Each light bulb is either on or off. You can turn the switches on and off as many times as you want, \
# but you can only enter the room one time to observe the light bulbs. \
# How can you figure out which switch corresponds to which light bulb?"
