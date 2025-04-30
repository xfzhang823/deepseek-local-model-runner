"""
batch_manager_async.py

Async-first batch manager for LLM tasks using `asyncio` and a unified request format.

Functions:
- `run_batch_tasks_async`: Execute mixed or uniform LLM tasks concurrently.
- `summarize_batch_async`: Batch summarization helper.
- `translate_batch_async`: Batch translation helper.
- `keyword_extraction_batch_async`: Batch keyword extraction helper.
- `topic_generation_batch_async`: Batch topic generation helper.
- `text_alignment_batch_async`: Batch text alignment helper.
"""

import asyncio
import logging
from typing import Any, Dict, List
from task_router_async import run_llm_task_async
from llm_response_models import LLMResponseBase

logger = logging.getLogger(__name__)


async def run_batch_tasks_async(
    batch_requests: List[Dict[str, Any]],
) -> List[LLMResponseBase]:
    """
    Run a mixed batch of LLM tasks concurrently.

    Uses `asyncio.gather` to dispatch each request in the same event loop.

    Args:
        batch_requests (List[Dict[str, Any]]):
            Each dict must include:
            - 'task_type' (str): LLM task name.
            - 'model' (str, optional): model identifier ('hf', 'awq', etc.).
            - 'mode' (str, optional): Sampling preset ('precision', 'balanced',
            'recall').
            - 'prompt_type' (str, optional): Template variant for keyword
            extraction.
            - 'kwargs' (dict): Task-specific parameters (e.g., {'text': ...,
            'target_lang': ...}).

    Returns:
        List[LLMResponseBase]: Pydantic response models in the order of requests.
    """
    tasks = []
    for req in batch_requests:
        task_type = req.get("task_type")
        model = req.get("model", "hf")
        mode = req.get("mode", "balanced")
        prompt_type = req.get("prompt_type", "default")
        params = req.get("kwargs", {})
        tasks.append(
            run_llm_task_async(
                task_type,
                model=model,
                mode=mode,
                prompt_type=prompt_type,
                **params,
            )
        )
    return await asyncio.gather(*tasks)


async def summarize_batch_async(
    texts: List[str], model: str = "hf", mode: str = "balanced"
) -> List[LLMResponseBase]:
    """
    Run summarization for a batch of texts asynchronously.

    Args:
        texts (List[str]): Input texts to summarize.
        model (str): model identifier ('hf', 'awq', etc.).
        mode (str): Sampling preset ('precision', 'balanced', 'creative').

    Returns:
        List[LLMResponseBase]: SummarizationResponse models containing summaries
        and metadata.
    """
    batch_requests = [
        {
            "task_type": "summarization",
            "model": model,
            "mode": mode,
            "kwargs": {"text": t},
        }
        for t in texts
    ]
    return await run_batch_tasks_async(batch_requests)


async def translate_batch_async(
    texts: List[str], target_lang: str, model: str = "hf", mode: str = "balanced"
) -> List[LLMResponseBase]:
    """
    Run translation for a batch of texts asynchronously.

    Args:
        texts (List[str]): Input texts to translate.
        target_lang (str): Language code for translation (e.g., 'French').
        model (str): Model identifier ('hf', 'awq', etc.).
        mode (str): Sampling preset ('precision', 'balanced', 'recall').

    Returns:
        List[LLMResponseBase]: TranslationResponse models containing translations
        and metadata.
    """
    batch_requests = [
        {
            "task_type": "translation",
            "model": model,
            "mode": mode,
            "kwargs": {"text": t, "target_lang": target_lang},
        }
        for t in texts
    ]
    return await run_batch_tasks_async(batch_requests)


async def keyword_extraction_batch_async(
    texts: List[str],
    model: str = "hf",
    mode: str = "balanced",
    prompt_type: str = "default",
) -> List[LLMResponseBase]:
    """
    Extract keywords from a batch of texts asynchronously.

    Args:
        texts (List[str]): Input texts for extraction.
        model (str): model identifier ('hf', 'awq', etc.).
        mode (str): Sampling preset ('precision', 'balanced', 'recall').
        prompt_type (str): Prompt template variant ('default', 'technical').

    Returns:
        List[LLMResponseBase]: KeywordExtractionResponse models containing keywords
        and metadata.
    """
    batch_requests = [
        {
            "task_type": "keyword_extraction",
            "model": model,
            "mode": mode,
            "prompt_type": prompt_type,
            "kwargs": {"text": t},
        }
        for t in texts
    ]
    return await run_batch_tasks_async(batch_requests)


async def topic_generation_batch_async(
    texts: List[str], model: str = "hf", mode: str = "balanced"
) -> List[LLMResponseBase]:
    """
    Generate high-level topics for a batch of texts asynchronously.

    Args:
        texts (List[str]): Input texts for topic generation.
        model (str): Model identifier ('hf', 'awq', etc.).
        mode (str): Sampling preset ('precision', 'balanced', 'recall').

    Returns:
        List[LLMResponseBase]: TopicGenerationResponse models containing topics and metadata.
    """
    batch_requests = [
        {
            "task_type": "topic_generation",
            "model": model,
            "mode": mode,
            "kwargs": {"text": t},
        }
        for t in texts
    ]
    return await run_batch_tasks_async(batch_requests)


async def text_alignment_batch_async(
    pairs: List[Dict[str, str]], model: str = "hf", mode: str = "balanced"
) -> List[LLMResponseBase]:
    """
    Run text alignment on a batch of source-target pairs asynchronously.

    Args:
        - pairs (List[Dict[str, str]]): List of dicts with 'source_text' and 'target_text'.
        - model (str): Model identifier ('hf', 'awq', etc.).
        - mode (str): Sampling preset ('precision', 'balanced', 'recall').

    Returns:
        List[LLMResponseBase]: TextAlignmentResponse models containing alignments
        and metadata.
    """
    batch_requests = [
        {
            "task_type": "text_alignment",
            "model": model,
            "mode": mode,
            "kwargs": {
                "source_text": p["source_text"],
                "target_text": p["target_text"],
            },
        }
        for p in pairs
    ]
    return await run_batch_tasks_async(batch_requests)
