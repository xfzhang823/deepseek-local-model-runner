"""
precompute_input_ids.py

Tokenize calibration texts into input_ids and save to disk.
Can be used as a script or imported as a function.
"""

import logging
import os
import torch
from pathlib import Path
from transformers import AutoTokenizer
from datasets import load_dataset
from typing import Optional, List, Union

logger = logging.getLogger(__name__)


# Config defaults
DEFAULT_MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
DEFAULT_CALIBRATION_DATASET = "wikitext"
DEFAULT_CALIBRATION_SUBSET = "wikitext-2-raw-v1"
DEFAULT_N_SAMPLES = 512
DEFAULT_MAX_SEQ_LEN = 2048
DEFAULT_OUTPUT_PATH = Path(
    "~/models/deepseek-awq-scrooge/calib_embeddings/calib_input_ids.pt"
).expanduser()


def load_calibration_texts(
    dataset_name: str = "wikitext",
    subset: Optional[str] = None,
    split: str = "train",
    n_samples: int = 512,
) -> List[str]:
    """Load and clean calibration texts."""
    logger.info(f"Loading dataset: {dataset_name}.")

    if dataset_name == "pileval":
        dataset = load_dataset("mit-han-lab/pile-val-backup", split="validation")
    elif dataset_name == "wikitext":
        dataset = load_dataset("wikitext", subset or "wikitext-2-raw-v1", split=split)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    # pyright: ignore[reportIndexIssue]
    texts = [t for t in dataset["text"] if isinstance(t, str) and t.strip()]
    import random

    random.shuffle(texts)
    return texts[:n_samples]


def precompute_input_ids(
    model_name: str = DEFAULT_MODEL_NAME,
    dataset_name: str = DEFAULT_CALIBRATION_DATASET,
    subset: Optional[str] = DEFAULT_CALIBRATION_SUBSET,
    n_samples: int = DEFAULT_N_SAMPLES,
    max_seq_len: int = DEFAULT_MAX_SEQ_LEN,
    output_path: Union[str, Path] = DEFAULT_OUTPUT_PATH,
) -> Path:
    """
    Tokenize calibration texts and save input_ids to disk.

    Args:
        model_name (str): Huggingface model name.
        dataset_name (str): Calibration dataset name ("pileval" or "wikitext").
        subset (Optional[str]): Subset name if needed (e.g., wikitext-2-raw-v1).
        n_samples (int): Number of texts to sample.
        max_seq_len (int): Max sequence length during tokenization.
        output_path (str or Path): Where to save input_ids.

    Returns:
        Path: The path where input_ids were saved.
    """
    output_path = Path(output_path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    texts = load_calibration_texts(
        dataset_name=dataset_name, subset=subset, n_samples=n_samples
    )

    logger.info(f"✅ Loaded {len(texts)} calibration texts.")

    encodings = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=max_seq_len,
        return_tensors="pt",
    )

    input_ids = encodings["input_ids"]
    torch.save(input_ids, output_path)

    logger.info(f"✅ Saved tokenized input_ids to: {output_path}")
    return output_path


if __name__ == "__main__":
    precompute_input_ids()
