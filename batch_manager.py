from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor
from api import run_llm_task
from llm_response_models import (
    LLMResponseBase,
    KeywordExtractionResponse,
    SummarizationResponse,
    TopicGenerationResponse,
    TextAlignmentResponse,
    TranslationResponse,
)
import logging


logger = logging.getLogger(__name__)


def run_batch_tasks(
    batch_requests: List[Dict[str, Any]], max_workers: int = 4
) -> List[LLMResponseBase]:
    """
    Run a batch of LLM tasks in parallel using thread-based concurrency.

    Args:
        batch_requests (List[Dict]): A list of dictionaries, each with:
            - 'task_type' (str): Type of the task (e.g., 'summarization', 'translation')
            - 'kwargs' (dict): Keyword arguments for that task
        max_workers (int): Maximum number of threads to use.

    Returns:
        List[Dict]: A list of task results (JSON-style dictionaries).
    """
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(run_llm_task, req["task_type"], **req["kwargs"])
            for req in batch_requests
        ]
        for future in futures:
            results.append(future.result())
    return results


def summarize_batch(
    texts: List[str], max_workers: int = 4
) -> List[SummarizationResponse]:
    """
    Run summarization for a batch of texts.

    Args:
        texts (List[str]): A list of text strings to summarize.
        max_workers (int): Number of threads to run in parallel.

    Returns:
        List[Dict]: Summaries and processing metadata.
    """
    batch_requests = [
        {"task_type": "summarization", "kwargs": {"text": t}} for t in texts
    ]
    return run_batch_tasks(batch_requests, max_workers)


def translate_batch(
    texts: List[str], target_lang: str = "French", max_workers: int = 4
) -> List[TranslationResponse]:
    """
    Run translation for a batch of texts into the target language.

    Args:
        texts (List[str]): A list of text strings to translate.
        target_lang (str): Language to translate into.
        max_workers (int): Number of threads to run in parallel.

    Returns:
        List[Dict]: Translated results and metadata.
    """
    batch_requests = [
        {"task_type": "translation", "kwargs": {"text": t, "target_lang": target_lang}}
        for t in texts
    ]
    return run_batch_tasks(batch_requests, max_workers)


def keyword_extraction_batch(
    texts: List[str], num_keywords: int = 5, max_workers: int = 4
) -> List[KeywordExtractionResponse]:
    """
    Extract keywords from a batch of texts.

    Args:
        texts (List[str]): List of texts to extract keywords from.
        num_keywords (int): Number of keywords to extract per text.
        max_workers (int): Number of threads to run in parallel.

    Returns:
        List[Dict]: Extracted keywords and metadata.
    """
    batch_requests = [
        {
            "task_type": "keyword_extraction",
            "kwargs": {"text": t, "num_keywords": num_keywords},
        }
        for t in texts
    ]
    return run_batch_tasks(batch_requests, max_workers)


def topic_generation_batch(
    texts: List[str], max_workers: int = 4
) -> List[TopicGenerationResponse]:
    """
    Generate topics for a batch of texts.

    Args:
        texts (List[str]): List of texts to generate topics from.
        max_workers (int): Number of threads to run in parallel.

    Returns:
        List[Dict]: Generated topics and metadata.
    """
    batch_requests = [
        {"task_type": "topic_generation", "kwargs": {"text": t}} for t in texts
    ]
    return run_batch_tasks(batch_requests, max_workers)


def text_alignment_batch(
    text_pairs: List[Dict[str, str]], max_workers: int = 4
) -> List[TextAlignmentResponse]:
    """
    Run text alignment on a list of source-target pairs.

    Args:
        - text_pairs (List[Dict[str, str]]): A list of dictionaries with 'source_text'
        and 'target_text'.
        - max_workers (int): Number of threads to run in parallel.

    Returns:
        List[Dict]: Alignment results for each pair.
    """
    batch_requests = [
        {
            "task_type": "text_alignment",
            "kwargs": {
                "source_text": pair["source_text"],
                "target_text": pair["target_text"],
            },
        }
        for pair in text_pairs
    ]
    return run_batch_tasks(batch_requests, max_workers)
