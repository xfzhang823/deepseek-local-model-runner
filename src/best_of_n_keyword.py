from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List
import logging

from key_word_extractor import KeywordExtractor
from model_loader import ModelLoader
from llm_response_models import KeywordExtractionResponse

logger = logging.getLogger(__name__)


def is_valid_keyword_response(resp: KeywordExtractionResponse) -> bool:
    return (
        resp.status == "success"
        and len(resp.keywords) >= 3
        and "[NO_KEYWORDS_FOUND]" not in resp.keywords
    )


def best_of_n_keyword_extraction(
    texts: List[str],
    n: int = 3,
    max_workers: int = 8,
    mode: str = "balanced",
    prompt_type: str = "default",
) -> List[KeywordExtractionResponse]:
    """
    Run best-of-N keyword extraction per text, selecting the first valid result.

    Args:
        texts (List[str]): Input documents.
        n (int): Number of sampling attempts per text.
        max_workers (int): Number of concurrent threads.
        mode (str): Sampling config name.
        prompt_type (str): Prompt template key.

    Returns:
        List[KeywordExtractionResponse]: Best result per input text.
    """
    tokenizer, model = ModelLoader.load_model()
    extractor = KeywordExtractor(model=model, tokenizer=tokenizer)

    def extract_with_retries(text: str) -> KeywordExtractionResponse:
        for attempt in range(n):
            result = extractor.extract(
                text=text,
                mode=mode,
                prompt_type=prompt_type,
            )
            if is_valid_keyword_response(result):
                logger.info(f"[✓] Valid result on attempt {attempt+1}")
                return result
        logger.warning("[×] No valid result after N tries — returning last.")
        return result  # Return last even if invalid

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(extract_with_retries, text) for text in texts]
        return [f.result() for f in as_completed(futures)]
