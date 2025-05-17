# utils/inference_benchmark.py
import time
import torch
from typing import Tuple
import logging

logger = logging.getLogger(__name__)


def measure_inference_speed(
    model,
    tokenizer,
    input_text: str,
    max_new_tokens: int = 50,
    device: str = "cuda",
) -> Tuple[float, float]:
    """Benchmark LLM inference speed (tokens/sec)."""
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    start_time = time.time()

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
        )

    elapsed = time.time() - start_time
    tokens_generated = len(outputs[0]) - len(inputs["input_ids"][0])
    tokens_per_sec = tokens_generated / elapsed

    logger.info(
        f"Generated {tokens_generated} tokens in {elapsed:.2f}s ({tokens_per_sec:.2f} tokens/sec)"
    )

    return elapsed, tokens_per_sec
