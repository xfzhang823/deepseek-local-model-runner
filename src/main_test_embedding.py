import os
from dotenv import load_dotenv
from transformers import AutoTokenizer
from awq import AutoAWQForCausalLM
import logging
import logging_config
from quantize.embed_batches_with_cache import embed_batches_with_cache

logger = logging.getLogger(__name__)


def main():
    # Define your input model path and save path

    load_dotenv()

    model_name = os.getenv("MODEL_NAME_HF")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoAWQForCausalLM.from_pretrained(
        model_name, device_map="auto", trust_remote_code=True
    )

    embed_batches_with_cache(
        model=model,
        tokenizer=tokenizer,
        batch_size=16,
        max_seq_len=2048,
    )


if __name__ == "__main__":
    main()
