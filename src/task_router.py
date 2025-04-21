"""
task_router.py

This module defines a central dispatch function for routing individual
LLM-related tasks (e.g., summarization, translation, keyword extraction)
to their corresponding handlers. It supports top-down model injection
by allowing tokenizer and model to be passed explicitly.

Each task function returns a structured Pydantic response model.
"""

from typing import Any, Dict
from llm_response_models import LLMResponseBase
from task_manager_async import (
    summarize_async,
    translate_async,
    extract_keywords_async,
    generate_topics_async,
    align_texts_async,
)


# Define once at the top of the module
ALLOWED_MODELS = {"hf", "awq", "openai-gpt4", "anthropic-claude"}


async def run_llm_task_async(
    task_type: str,
    *,  # end of poistional parameters
    model: str = "hf",
    mode: str = "balanced",
    prompt_type: str = "default",
    **kwargs: Any,
) -> LLMResponseBase:
    """
    Dispatch a single LLM task asynchronously to its handler.

    Args:
        - task_type (str): One of 'summarization', 'translation',
                         'keyword_extraction', 'topic_generation',
                         'text_alignment'.
        - model (str): LLM backend identifier (e.g., 'hf', 'awq').
        - mode (str): Sampling preset for the task ('precision', 'balanced', 'recall').
        - prompt_type (str): Prompt template variant (only for keyword extraction).
        **kwargs: Additional parameters for the task (e.g., text, target_lang).

    Returns:
        LLMResponseBase: Pydantic model with the task result.

    Raises:
        ValueError: If task_type is not supported.

    """
    # 1) Enforce a whitelist
    if model not in ALLOWED_MODELS:
        raise ValueError(f"Unsupported model: {model}")

    # 2) Enforce valid modes, too (optional)
    if mode not in {"precision", "balanced", "recall"}:
        raise ValueError(f"Unsupported sampling mode: {mode}")

    # 3) Dispatch as before
    handler = {
        "summarization": summarize_async,
        "translation": translate_async,
        "keyword_extraction": extract_keywords_async,
        "topic_generation": generate_topics_async,
        "text_alignment": align_texts_async,
    }.get(task_type)

    if handler is None:
        raise ValueError(f"Unsupported task: {task_type}")

    return await handler(model=model, mode=mode, prompt_type=prompt_type, **kwargs)
