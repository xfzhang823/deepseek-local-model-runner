"""
utils/calib_data.py

Custom function to override AutoAWQ's official get_calib_dataset function.

Uses sentence-aware chunking to avoid mid-sentence truncation,
and adds padding with a minimal cutoff to ensure usable activation stats.

Requires:
    import nltk
    nltk.download('punkt')
"""

import logging
from typing import List, Union
import torch
from datasets import load_dataset
import nltk
from nltk.tokenize.punkt import PunktSentenceTokenizer

# Ensure punkt tokenizer is available
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

# Initialize once to avoid triggering language detection
sentence_tokenizer = PunktSentenceTokenizer()

logger = logging.getLogger(__name__)


def get_calib_dataset_sent_aware(
    data: Union[str, List[str]] = "pileval",
    tokenizer=None,
    n_samples: int = 128,
    max_seq_len: int = 512,
    split: str = "train",
    text_column: str = "text",
    min_seq_len: int = 64,
) -> List[torch.Tensor]:
    """
    Sentence-aware calibration dataset loader for AWQ-style quantization.
    Returns a list of tokenized samples with shape [1, max_seq_len], padded as needed.

    Args:
        data: HuggingFace dataset name or list of raw strings.
        tokenizer: Tokenizer to encode text.
        n_samples: Total number of usable samples to return.
        max_seq_len: Max number of tokens per sample.
        split: Dataset split (e.g., "train", "validation").
        text_column: Name of the column containing raw text.
        min_seq_len: Minimum real token count to keep a sample (before padding).

    Returns:
        List[torch.Tensor]: Each sample is shape [1, max_seq_len]
    """
    if tokenizer is None:
        raise ValueError("Tokenizer must be provided to get_calib_dataset()")

    pad_id = tokenizer.pad_token_id or 0

    if isinstance(data, str):
        if data == "pileval":
            try:
                dataset = load_dataset(
                    "mit-han-lab/pile-val-backup",
                    split="validation",
                    trust_remote_code=True,
                )
            except Exception as e:
                logger.warning(
                    f"⚠️ Failed to load 'pileval': {e}. Falling back to 'c4:en'."
                )
                dataset = load_dataset(
                    "c4", "en", split="validation", trust_remote_code=True
                )
        else:
            dataset = load_dataset(data, split=split, trust_remote_code=True)

        dataset = dataset.shuffle(seed=42)

    elif isinstance(data, list) and all(isinstance(x, str) for x in data):
        dataset = [{text_column: x} for x in data]
    else:
        raise TypeError(
            "`data` must be either a dataset name (str) or a list of strings."
        )

    samples = []
    for entry in dataset:
        if not isinstance(entry, dict) or text_column not in entry:
            logger.warning(f"⚠️ Skipping malformed entry: {entry}")
            continue

        line = entry[text_column].strip()
        sentences = sentence_tokenizer.tokenize(line)
        current_tokens = []

        for sent in sentences:
            sent_tokens = tokenizer.encode(sent, add_special_tokens=False)

            if len(current_tokens) + len(sent_tokens) <= max_seq_len:
                current_tokens.extend(sent_tokens)
            else:
                if len(current_tokens) >= min_seq_len:
                    padded = current_tokens + [pad_id] * (
                        max_seq_len - len(current_tokens)
                    )
                    samples.append(torch.tensor([padded]))
                current_tokens = sent_tokens if len(sent_tokens) <= max_seq_len else []

            if len(samples) >= n_samples:
                break

        # Add final block if it's large enough
        if (
            min_seq_len <= len(current_tokens) < max_seq_len
            and len(samples) < n_samples
        ):
            padded = current_tokens + [pad_id] * (max_seq_len - len(current_tokens))
            samples.append(torch.tensor([padded]))

        if len(samples) >= n_samples:
            break

    logger.info(f"✅ Collected {len(samples)} sentence-aware calibration samples")
    return samples
