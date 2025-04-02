"""
TBA
"""

from typing import Any, Dict, List
import logging

# User defined
from task_router import run_llm_task
from batch_manager import (
    run_batch_tasks,
    summarize_batch,
    translate_batch,
    keyword_extraction_batch,
    topic_generation_batch,
    text_alignment_batch,
)


logger = logging.getLogger(__name__)


def run_llm_batch_task(
    batch_requests: List[Dict[str, Any]], max_workers: int = 4
) -> List[Dict[str, Any]]:
    """
    Run a batch of tasks concurrently.

    Args:
        batch_requests (List[Dict]): A list of task request dictionaries.
        max_workers (int): Number of parallel threads.

    Returns:
        List[Dict]: List of task results.
    """
    return run_batch_tasks(batch_requests, max_workers=max_workers)


# Expose convenience batch functions directly
__all__ = [
    "run_llm_task",
    "run_llm_batch_task",
    "summarize_batch",
    "translate_batch",
    "keyword_extraction_batch",
    "topic_generation_batch",
    "text_alignment_batch",
]
