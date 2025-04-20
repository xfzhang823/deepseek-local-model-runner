# llm_sampling_utils.py
from concurrent.futures import ThreadPoolExecutor
from typing import Callable
import logging

logger = logging.getLogger(__name__)


def sample_best_of_n(
    extract_fn: Callable[[], object],
    n: int = 4,
    max_workers: int = 4,
    scoring_fn: Callable[[object], float] = None,
):
    """
    Runs extract_fn n times in parallel and returns the best result using scoring_fn.

    Args:
        extract_fn: Function that returns a response object (e.g., KeywordExtractionResponse)
        n: Number of samples to generate
        max_workers: Number of threads to use
        scoring_fn: Function to evaluate quality (higher is better)

    Returns:
        Best result (based on score or first success)
    """
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(extract_fn) for _ in range(n)]
        results = [f.result() for f in futures]

    successful = [r for r in results if r.status == "success"]

    if not successful:
        logger.warning("All sampling attempts failed.")
        return results[0]  # return first error

    if scoring_fn:
        successful.sort(key=scoring_fn, reverse=True)

    return successful[0]
