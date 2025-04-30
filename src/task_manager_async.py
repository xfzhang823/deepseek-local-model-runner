"""
task_manager_async.py

* Async versions of the existing LLM task functions.

Each function in this module wraps a blocking task implementation
(from `_sync_tasks.py`) using `asyncio.run_in_executor` to allow
non-blocking concurrency with `async def` entrypoints.

The signatures accept dynamic **kwargs so they can be invoked directly
from a schema-based router (`TaskRequestModel`). These kwargs are passed
along to the underlying sync implementations such as `summarize(...)`,
`translate(...)`, etc.

Usage:
    await summarize_async(model="hf", mode="balanced", text="...")
"""

import asyncio
import logging
from typing import Any, Awaitable, Callable, TypeVar
from functools import partial, wraps
from llm_response_models import (
    SummarizationResponse,
    TranslationResponse,
    KeywordExtractionResponse,
    TopicGenerationResponse,
    TextAlignmentResponse,
)
from _sync_tasks import (
    summarize as _summarize_sync,
    translate as _translate_sync,
    extract_keywords as _extract_keywords_sync,
    generate_topics as _generate_topics_sync,
    align_texts as _align_texts_sync,
)

logger = logging.getLogger(__name__)

# Import blocking implementations from the sync task module
T = TypeVar("T")


def sync_to_async(fn: Callable[..., T]) -> Callable[..., Awaitable[T]]:
    """
    Decorator: wraps a blocking function so it can be awaited in asyncio.

    All args and kwargs are captured via functools.partial and executed
    in the event loop's default executor (threadpool).
    """

    async def wrapper(*args: Any, **kwargs: Any) -> T:
        loop = asyncio.get_running_loop()
        task = partial(fn, *args, **kwargs)

        logger.info(
            f"[Async] Executing {fn.__name__} in executor with args={args} kwargs={kwargs}"
        )

        try:
            return await loop.run_in_executor(None, task)
        except Exception as e:
            logger.exception(f"[Async] Exception in {fn.__name__}: {e}")
            raise

    return wrapper


@sync_to_async
def summarize_async(**kwargs) -> SummarizationResponse:
    """
    Asynchronously summarize text using the sync summarization backend.
    Expected kwargs:
        - text (str)
        - mode (str)
        - model (str)
    """
    logger.debug("[Async] Running summarize_async...")
    return _summarize_sync(**kwargs)


@sync_to_async
def translate_async(**kwargs) -> TranslationResponse:
    """
    Asynchronously translate text using the sync translation backend.
    Expected kwargs:
        - text (str)
        - target_lang (str)
        - mode (str)
        - model (str)
    """
    logger.debug("[Async] Running translate_async...")
    return _translate_sync(**kwargs)


@sync_to_async
def extract_keywords_async(**kwargs) -> KeywordExtractionResponse:
    """
    Asynchronously extract keywords using the sync keyword extraction backend.
    Expected kwargs:
        - text (str)
        - mode (str)
        - model (str)
        - prompt_type (str)
    """
    logger.debug("[Async] Running extract_keywords_async...")
    return _extract_keywords_sync(**kwargs)


@sync_to_async
def generate_topics_async(**kwargs) -> TopicGenerationResponse:
    """
    Asynchronously generate topics using the sync backend.
    Expected kwargs:
        - text (str)
        - model (str)
        - mode (str)
    """
    logger.debug("[Async] Running generate_topics_async...")
    return _generate_topics_sync(**kwargs)


@sync_to_async
def align_texts_async(**kwargs) -> TextAlignmentResponse:
    """
    Asynchronously align text pairs using the sync backend.
    Expected kwargs:
        - source_text (str)
        - target_text (str)
        - model (str)
        - mode (str)
    """
    logger.debug("[Async] Running align_texts_async...")
    return _align_texts_sync(**kwargs)
