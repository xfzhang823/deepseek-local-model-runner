"""main_quant_model_analyze.py"""

from pathlib import Path
import logging
from utils.compare_quant_models import compare_tensors
from project_config import (
    DEEPSEEK_R1_DISTILL_QUANT_MODEL_DIR,
    DEEPSEEK_R1_DISTILL_QUANT_MODEL_OUTPUT_DIR,
)
import logging_config

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    casper_model_dir = Path(
        "~/models/casperhansen-deepseek-r1-distill-qwen-1.5b-awq"
    ).expanduser()
    casper_tensor_file = casper_model_dir / "model.safetensors"
    # scrooge_tensor_file = DEEPSEEK_R1_DISTILL_QUANT_MODEL_DIR / "model.safetensors"
    scrooge_tensor_file = (
        DEEPSEEK_R1_DISTILL_QUANT_MODEL_OUTPUT_DIR
        / "quantized_layers"
        / "model.layers.0.self_attn.k_proj.pt"
    )
    save_dir = Path(
        "~/dev/deepseek_local_runner/documents/model_comparison/casper_scrooge_comparison"
    ).expanduser()

    attn_layer_key = "model.layers.0.self_attn.k_proj"

    gate_layer_key = "model.layers.0.mlp.gate_proj"

    compare_tensors(
        model_1_file=casper_tensor_file,
        model_2_file=scrooge_tensor_file,
        tensor_key_prefix=attn_layer_key,
        save_dir=save_dir,
        model_1_name="casper model",
        model_2_name="scrooge model",
    )

    # Plot zero points if extracted separately
    # compare_zero_plots(their_zero_tensor, your_zero_tensor)
