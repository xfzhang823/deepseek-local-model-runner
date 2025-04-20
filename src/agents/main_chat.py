"""testing chat"""

import os
import datetime
import logging
from transformers import pipeline
from dotenv import load_dotenv
import logging_config

# Setup logging
logger = logging.getLogger(__name__)

# Time tracking
start_time = datetime.datetime.now()
logger.info(f"Start time: {start_time.strftime('%Y-%m_%d %H:%M:%S')}")


# Model name from .env
load_dotenv()
model_name = os.getenv("MODEL_NAME")

logger.info("Loading pipeline...")
pipe = pipeline("text-generation", model=model_name)
logger.info("Pipeline loaded.")

query = "Where is the capital of China?"
# query = "Why might a submarine need artificial lights even though it's surrounded by water that conducts light?"
# query = "Imagine an alien race is telepathic, will their ML system primarily reducing dimensionality or \
# expanding it?"
# query = "There are three switches, but they are not labelled. Each switch corresponds to one of three light bulbs in a room. \
# Each light bulb is either on or off. You can turn the switches on and off as many times as you want, \
# but you can only enter the room one time to observe the light bulbs. \
# How can you figure out which switch corresponds to which light bulb?"
messages = pipe(query, max_new_tokens=1000, truncation=True)
logger.info("Generating output...")
logger.info(messages)

# Time tracking
end_time = datetime.datetime.now()
logger.info(f"End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")

elapsed_time = end_time - start_time
logger.info(f"Elapsed Time: {elapsed_time}")
