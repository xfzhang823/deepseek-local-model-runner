"""
llm_inference.py

Centralized, sync LLM inference logic with configurable model.

This module exposes two core functions:

- `generate(prompt: str, config: GenerationConfig, model: str = "hf")`
  Runs a single-pass generation.

- `generate_with_thinking(prompt: str, config: GenerationConfig,
                         thinking_tags: tuple = ("<think>","</think>"),
                         answer_tags: tuple = ("<answer>","</answer>"),
                         model: str = "hf")`
  Wraps your prompt in reasoning tags and returns (thinking, answer).

Supported models:
- `hf`: Hugging Face 4/8-bit quantized models
(via `hf_loader.HF_ModelLoader`).
- `awq`: AWQ-quantized models (via `awq_loader.AWQ_ModelLoader`).
"""

import time
from typing import Tuple
import torch

from schemas.generation_config import GenerationConfig, TaskType
from loaders.hf_loader import HF_ModelLoader
from loaders.awq_loader import AWQ_ModelLoader
from artifacts import write_artifact


def _get_loader(model: str):
    """Pick the model loader based on model name."""
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
    tokenizer, lm = _get_loader(model).load_model()

    # Tokenize & move to device
    inputs = tokenizer(prompt, return_tensors="pt")
    device = next(lm.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Generate
    start = time.time()
    with torch.no_grad():
        outputs = lm.generate(
            **inputs,
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature,
            top_k=config.top_k,
            top_p=config.top_p,
        )
    elapsed = time.time() - start
    response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    result = {
        "response_text": response_text,
        "processing_time": time.strftime("%H:%M:%S", time.gmtime(elapsed)),
    }

    # Write artifact
    # Guess task_type from prompt tags (you can adjust as needed)
    task = (
        TaskType.SUMMARIZATION.value
        if "<summary>" in prompt
        else TaskType.TRANSLATION.value
    )
    write_artifact(
        task_type=task,
        input_params={"prompt": prompt, **config.__dict__},
        config_obj=config,
        output_obj=result,
    )

    return result


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
        - prompt: User question (raw).
        - config: Sampling settings.
        - thinking_tags: Tags to wrap reasoning steps.
        - answer_tags: Tags to wrap final answer.
        - model: 'hf' or 'awq'.

    Returns:
        (thinking_steps, final_answer)
    """
    start_think, end_think = thinking_tags
    start_ans, end_ans = answer_tags

    # Wrap prompt
    wrapped = f"{start_think}\n{prompt}\n{end_think}\n{start_ans}"

    # Call standard generate
    start = time.time()
    raw = generate(wrapped, config, model)["response_text"]
    elapsed = time.time() - start

    # Helper to extract between tags
    def _extract(text: str, tag_start: str, tag_end: str) -> str:
        parts = text.split(tag_start)
        if len(parts) > 1 and tag_end in parts[1]:
            return parts[1].split(tag_end)[0].strip()
        return ""

    thinking = _extract(raw, start_think, end_think)
    answer = _extract(raw, start_ans, end_ans) or raw.replace(thinking, "").strip()

    artifact_payload = {
        "thinking": thinking,
        "answer": answer,
        "processing_time": time.strftime("%H:%M:%S", time.gmtime(elapsed)),
    }

    # Write artifact (adjust task_type logic as needed)
    task = (
        TaskType.TEXT_ALIGNMENT.value
        if "<alignment>" in prompt
        else TaskType.TOPIC_GENERATION.value
    )
    write_artifact(
        task_type=task,
        input_params={"prompt": prompt, **config.__dict__},
        config_obj=config,
        output_obj=artifact_payload,
    )

    return thinking, answer
