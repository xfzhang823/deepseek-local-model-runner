"""
embed_batches_with_cache.py

Utility to tokenize (via precompute module), then embed input_ids in batches,
with crash-safe partial caching and timing.
"""

import time
import logging
import os
import pickle
from dotenv import load_dotenv
from typing import List, Optional, Union
from pathlib import Path

import torch
from torch import nn
from transformers import PreTrainedTokenizerBase
from tqdm import tqdm

# 🧩 Import your external tokenizer precompute function
from quantize.precompute_input_ids import precompute_input_ids

logger = logging.getLogger(__name__)

EMBEDDINGS_FILE = Path(
    "~/models/deepseek-awq-scrooge/calib_embeddings/full_embeddings.pt"
).expanduser()
EMBEDDINGS_CACHE_FILE = Path(
    "~/models/deepseek-awq-scrooge/calib_embeddings/embeddings_cache.pkl"
).expanduser()


def embed_batches_with_cache(
    model: nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    batch_size: int = 16,
    max_seq_len: int = 2048,
    cache_path: Union[str, Path] = EMBEDDINGS_CACHE_FILE,
    overwrite_cache: bool = False,
    embeds_path: Union[str, Path] = EMBEDDINGS_FILE,
) -> torch.Tensor:
    """
    Embed calibration texts in batches, saving partial results after each batch.

    Args:
        model (nn.Module): HuggingFace model with `.embed_tokens` layer.
        tokenizer (PreTrainedTokenizerBase): Tokenizer matching the model.
        batch_size (int): Mini-batch size for embedding.
        max_seq_len (int): Max sequence length for tokenization.
        cache_path (Union[str, Path]): Where to store partial results.
        overwrite_cache (bool): If True, starts fresh even if cache exists.
        embeds_path (Optional[str or Path]): If provided, save final embeddings.

    Returns:
        torch.Tensor: Full stacked embeddings [n_samples, max_seq_len, hidden_dim].

    [Raw Texts]
    ↓
    ╭─────────────────────────────────────────────╮
    │   Tokenizer (from HuggingFace)              │
    │   - Text → Input IDs (integers)             │
    ╰─────────────────────────────────────────────╯
        ↓
    [Input IDs Tensor]
        ↓
    ╭──────────────────────────────────────────────╮
    │   Batch Loop (batch_size = 16)               │
    │                                              │
    │   For each batch:                            │
    │     1. Move Input IDs → GPU                  │
    │     2. Pass into Model’s Embedding Layer     │
    │        - model.embed_tokens(input_ids)       │
    │        - OR model.transformer.wte(input_ids) │
    │     3. Get [batch_size, seq_len, hidden_dim] │
    │     4. Save batch embeddings to list         │
    │     5. Update Partial Cache                  │
    ╰──────────────────────────────────────────────╯
        ↓
    [All Embedded Batches Collected]
        ↓
    ╭──────────────────────────────────────────────────╮
    │   Stack Batches into Final Tensor                │
    │   - Shape: [n_samples, max_seq_len, hidden_dim]  │
    ╰──────────────────────────────────────────────────╯
        ↓
    [Final Full Embeddings Tensor] ✅


    Note: Where is the model's embedding layer
    model (Qwen2AWQForCausalLM wrapper)
    └── model (the real transformer model)
            └── get_input_embeddings()  ← (correct API to access embedding weights)
    * Use standard hugging face .get_input_embeddings()
    """
    cache_path = Path(cache_path).expanduser()
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    # Step 1: Precompute or load cached input_ids
    start_time = time.time()
    print("🚀 Preparing input_ids...")

    load_dotenv()
    model_name = os.getenv("MODEL_NAME_HF")
    if model_name is None:
        raise ValueError("MODEL_NAME_HF environment variable is not set")

    # Default input_ids path (same as used inside precompute_input_ids)
    default_input_ids_path = Path(
        "~/models/deepseek-awq-scrooge/calib_embeddings/calib_input_ids.pt"
    ).expanduser()

    if default_input_ids_path.exists():
        print(f"✅ Found cached input_ids at {default_input_ids_path}")
        input_ids_file = default_input_ids_path
    else:
        print("🔄 input_ids not found — generating now...")
        input_ids_file = precompute_input_ids(
            model_name=model_name,
            dataset_name="pileval",
            max_seq_len=max_seq_len,
        )

    end_tokenize = time.time()
    logger.info(f"✅ input_ids ready. Time elapsed: {end_tokenize - start_time:.2f}s")

    # Load from input ids file
    input_ids = torch.load(input_ids_file)  # ← manually load

    n_texts = input_ids.size(0)
    logger.info(f"📈 Total samples: {n_texts}")

    # Step 2: Embedding
    if cache_path.exists() and not overwrite_cache:
        print(f"🔄 Loading partial cache from {cache_path}")
        with cache_path.open("rb") as f:
            cached_data = pickle.load(f)
        embedded_batches = cached_data["embedded_batches"]
        completed = cached_data["completed_indices"]
    else:
        print(f"🚀 Starting fresh embedding generation...")
        embedded_batches = []
        completed = set()

    device = next(model.parameters()).device
    model.eval()

    # logger.info(f"Model class: {model.__class__}")
    # logger.info(f"Model attributes: {dir(model)}")
    # logger.info(f"Inner model attributes: {dir(model.model)}")

    for start_idx in tqdm(range(0, n_texts, batch_size), desc="Embedding batches"):
        end_idx = min(start_idx + batch_size, n_texts)

        if all(idx in completed for idx in range(start_idx, end_idx)):
            continue

        batch_input_ids = input_ids[start_idx:end_idx].to(device)

        with torch.no_grad():
            try:
                embedding_layer = model.model.get_input_embeddings()
            except AttributeError:
                raise AttributeError(
                    "Model does not have a get_input_embeddings method."
                )

            embeddings = embedding_layer(batch_input_ids)

        embedded_batches.append(embeddings.cpu())

        for idx in range(start_idx, end_idx):
            completed.add(idx)

        # Save partial cache
        with cache_path.open("wb") as f:
            pickle.dump(
                {"embedded_batches": embedded_batches, "completed_indices": completed},
                f,
            )

    final_embeds = torch.cat(embedded_batches, dim=0)

    # Save the final full tensor
    final_save_path = embeds_path
    torch.save(final_embeds, final_save_path)
    logger.info(f"💾 Full embeddings tensor saved to {final_save_path}")

    end_total_time = time.time()
    logger.info(f"🏁 Total embedding pipeline time: {end_total_time - start_time:.2f}s")

    return final_embeds


if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    from transformers import AutoTokenizer
    from awq import AutoAWQForCausalLM

    load_dotenv()

    model_name = os.getenv("MODEL_NAME_HF")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoAWQForCausalLM.from_quantized(
        model_name, device_map="auto", trust_remote_code=True
    )

    embed_batches_with_cache(
        model=model,
        tokenizer=tokenizer,
        batch_size=16,
        max_seq_len=2048,
    )
