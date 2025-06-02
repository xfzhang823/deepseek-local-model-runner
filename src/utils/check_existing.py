import os
from pathlib import Path
import logging
from typing import Optional, List
import torch.nn as nn
from awq.utils.module import get_named_linears
from awq.utils.module import exclude_layers_to_not_quantize


def are_block_sublayers_quantized(
    block_module: nn.Module,
    block_idx: int,
    save_dir: str,
    modules_to_not_convert: Optional[List[str]] = None,
    prefix: str = "model.layers",
) -> bool:
    """
    Check if all quantized .pt files exist for each Linear sublayer in a block,
    excluding those in the `exclude` list.

    Args:
        block_module (nn.Module): The transformer block (e.g., model.layers[3]).
        block_idx (int): Index of the block.
        save_dir (str): Directory where .pt files are saved.
        prefix (str): File prefix (default: 'model.layers').
        expected_layers (Optional[list[str]]): If provided, only check these names.
        exclude (Optional[list[str]]): Layer suffixes to skip (e.g., ['o_proj', 'lm_head']).

    Returns:
        bool: True if all required sublayer .pt files are present, False otherwise.
    """
    named_linears = exclude_layers_to_not_quantize(
        get_named_linears(block_module), modules_to_not_convert
    )
    sublayer_names = list(named_linears.keys())

    all_exist = True

    for name in sublayer_names:
        filename = f"{prefix}.{block_idx}.{name}.pt"
        file_path = os.path.join(save_dir, filename)
        if not os.path.exists(file_path):
            logging.warning(f"🟥 Missing: {filename}")
            all_exist = False
        else:
            logging.debug(f"✅ Found: {filename}")

    if all_exist:
        logging.info(f"✅ All sublayers for Block {block_idx} are quantized.")
    else:
        logging.info(f"🔧 Incomplete quantization for Block {block_idx}.")

    return all_exist
