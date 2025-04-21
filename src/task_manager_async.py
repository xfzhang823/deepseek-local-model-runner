"""
task_manager_async.py

* Async versions of the existing LLM task functions.

Uses a simple decorator to convert blocking sync functions into async callables
via `asyncio.get_running_loop().run_in_executor'.
"""

import asyncio
import logging
from llm_response_models import (
    SummarizationResponse,
    TranslationResponse,
    KeywordExtractionResponse,
    TopicGenerationResponse,
    TextAlignmentResponse,
)
from project_config import get_config
from loaders.dispatch_loader import get_model_loader  # loads tokenizer & model

# Import blocking implementations from the sync task module
from _sync_tasks import (
    summarize as _summarize_sync,
    translate as _translate_sync,
    extract_keywords as _extract_keywords_sync,
    generate_topics as _generate_topics_sync,
    align_texts as _align_texts_sync,
)

logger = logging.getLogger(__name__)


def sync_to_async(fn):
    """
    Decorator: wraps a blocking function so it can be awaited in asyncio.
    """

    async def wrapper(*args, **kwargs):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, fn, *args, **kwargs)

    return wrapper


@sync_to_async
async def summarize_async(text: str, mode: str = "balanced") -> SummarizationResponse:
    """
    Async summarization: maps directly to the sync `summarize` function.
    """
    return _summarize_sync(text, mode)


@sync_to_async
async def translate_async(
    text: str, target_lang: str = "French", mode: str = "balanced"
) -> TranslationResponse:
    """
    Async translation: maps directly to the sync `translate` function.
    """
    return _translate_sync(text, target_lang, mode)


@sync_to_async
async def extract_keywords_async(
    text: str, mode: str = "balanced"
) -> KeywordExtractionResponse:
    """
    Async keyword extraction: maps directly to the sync `extract_keywords` function.
    """
    return _extract_keywords_sync(text, mode)


@sync_to_async
async def generate_topics_async(text: str) -> TopicGenerationResponse:
    """
    Async topic generation: maps directly to the sync `generate_topics` function.
    """
    return _generate_topics_sync(text)


@sync_to_async
async def align_texts_async(
    source_text: str, target_text: str
) -> TextAlignmentResponse:
    """
    Async text alignment: maps directly to the sync `align_texts` function.
    """
    return _align_texts_sync(source_text, target_text)


logger = logging.getLogger(__name__)


def sync_to_async(fn):
    """
    Decorator: wraps a blocking function so it can be awaited in asyncio.
    """

    async def wrapper(*args, **kwargs):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, fn, *args, **kwargs)

    return wrapper


@sync_to_async
async def summarize_async(text: str, mode: str = "balanced") -> SummarizationResponse:
    """
    Async summarization: maps directly to the sync `summarize` function.
    """
    return _summarize_sync(text, mode)


@sync_to_async
async def translate_async(
    text: str, target_lang: str = "French", mode: str = "balanced"
) -> TranslationResponse:
    """
    Async translation: maps directly to the sync `translate` function.
    """
    return _translate_sync(text, target_lang, mode)


@sync_to_async
async def extract_keywords_async(
    text: str, mode: str = "balanced"
) -> KeywordExtractionResponse:
    """
    Async keyword extraction: maps directly to the sync `extract_keywords` function.
    """
    return _extract_keywords_sync(text, mode)


@sync_to_async
async def generate_topics_async(text: str) -> TopicGenerationResponse:
    """
    Async topic generation: maps directly to the sync `generate_topics` function.
    """
    return _generate_topics_sync(text)


@sync_to_async
async def align_texts_async(
    source_text: str, target_text: str
) -> TextAlignmentResponse:
    """
    Async text alignment: maps directly to the sync `align_texts` function.
    """
    return _align_texts_sync(source_text, target_text)
