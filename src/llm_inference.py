"""
llm_inference.py

Centralized, sync LLM inference logic with configurable backends.

This module exposes two core functions:

- `generate(prompt: str, config: GenerationConfig, backend: str = "hf")`
  Runs a single-pass generation.

- `generate_with_thinking(prompt: str, config: GenerationConfig, \
                         thinking_tags: tuple = ("<think>","</think>"), \
                         answer_tags: tuple = ("<answer>","</answer>"), \
                         backend: str = "hf")`
  Wraps your prompt in reasoning tags and returns (thinking, answer).

Supported backends:
- `hf`: Hugging Face 4/8-bit quantized models (via `hf_loader.HF_ModelLoader`).
- `awq`: AWQ-quantized models (via `awq_loader.AWQ_ModelLoader`).

Under the hood, each call:
1. Selects the appropriate loader.
2. Tokenizes and moves inputs to the model device.
3. Performs `.generate(...)` with settings from `GenerationConfig`.
4. Decodes and returns outputs.
"""

import time
from typing import Tuple
import torch
from project_config import GenerationConfig
from loaders.hf_loader import HF_ModelLoader
from loaders.awq_loader import AWQ_ModelLoader


def _get_loader(model: str):
    """Pick the model loader based on backend name."""
    if model.lower() == "awq":
        return AWQ_ModelLoader
    return HF_ModelLoader


def generate(prompt: str, config: GenerationConfig, model: str = "hf") -> dict:
    """
    Run a standard LLM generation.

    Args:
        - prompt: Input text.
        - config: Sampling configuration.
        - model: 'hf' or 'awq'.

    Returns:
        Dict containing:
          - 'response_text': decoded string
          - 'processing_time': formatted time string
    """

    # Load model & tokenizer
    tokenizer, model = _get_loader(model).load_model()

    # Tokenize & move to device
    inputs = tokenizer(prompt, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Generate
    start = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature,
            top_k=config.top_k,
            top_p=config.top_p,
        )
    elapsed = time.time() - start
    response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return {
        "response_text": response_text,
        "processing_time": time.strftime("%H:%M:%S", time.gmtime(elapsed)),
    }


def generate_with_thinking(
    prompt: str,
    config: GenerationConfig,
    thinking_tags: Tuple[str, str] = ("<think>", "</think>"),
    answer_tags: Tuple[str, str] = ("<answer>", "</answer>"),
    model: str = "hf",
) -> Tuple[str, str]:
    """
    Run LLM generation with explicit reasoning and answer separation.

    Args:
        prompt: User question (raw).
        config: Sampling settings.
        thinking_tags: Tags to wrap reasoning steps.
        answer_tags: Tags to wrap final answer.
        backend: 'hf' or 'awq'.

    Returns:
        (thinking_steps, final_answer)
    """
    # Prepare the wrapped prompt
    start_think, end_think = thinking_tags
    start_ans, end_ans = answer_tags
    wrapped = f"{start_think}\n{prompt}\n{end_think}\n{start_ans}"

    # Call the standard generate
    raw = generate(wrapped, config, model)["response_text"]

    # Parse sections
    def _extract(text: str, start: str, end: str) -> str:
        parts = text.split(start)
        if len(parts) > 1 and end in parts[1]:
            return parts[1].split(end)[0].strip()
        return ""

    thinking = _extract(raw, start_think, end_think)
    answer = _extract(raw, start_ans, end_ans) or raw.replace(thinking, "").strip()
    return thinking, answer
