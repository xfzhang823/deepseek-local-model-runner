from pathlib import Path
import torch
import os
import logging
from awq.utils.module import get_named_linears
from quantize.scrooge_awq_quantizer import ScroogeAwqQuantizer
from pipelines.scrooge_quant_pipeline import load_model_and_tokenizer
from project_config import DEEPSEEK_R1_DISTILL_QUANT_MODEL_OUTPUT_DIR
import logging_config
import nltk

nltk.download("punkt")

logger = logging.getLogger(__name__)

# # Currently quantizing at this level
# SAVE_DIR = DEEPSEEK_R1_DISTILL_QUANT_MODEL_OUTPUT_DIR / "activation_linear_inputs"
# N_SAMPLES = 96
# SEQ_LEN = 512

# # Try this
# SAVE_DIR = (
#     DEEPSEEK_R1_DISTILL_QUANT_MODEL_OUTPUT_DIR / "activation_linear_inputs_128_samples"
# )
# N_SAMPLES = 128
# SEQ_LEN = 512

# # Try this
# SAVE_DIR = (
#     DEEPSEEK_R1_DISTILL_QUANT_MODEL_OUTPUT_DIR / "activation_linear_inputs_192_samples"
# )
# N_SAMPLES = 192
# SEQ_LEN = 256

# Try this
SAVE_DIR = (
    DEEPSEEK_R1_DISTILL_QUANT_MODEL_OUTPUT_DIR / "activation_linear_inputs_256_samples"
)
N_SAMPLES = 256
SEQ_LEN = 256

quant_config = {
    "w_bit": 4,
    "q_group_size": 128,
    "zero_point": True,
    "version": "GEMM",
    "quant_method": "awq",
}


def save_all_block_input_activations():
    os.makedirs(SAVE_DIR, exist_ok=True)

    model, tokenizer = load_model_and_tokenizer(
        base_model="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    )

    quantizer = ScroogeAwqQuantizer(
        model=model,
        tokenizer=tokenizer,
        quant_config=quant_config,
        max_calib_samples=N_SAMPLES,
        max_calib_seq_len=SEQ_LEN,
    )

    # === Step 1: Run calibration init ===
    # 🚨 This internally uses GPU to run the first forward pass
    quantizer.init_calibration()

    # === Step 2: Move everything to consistent device ===
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 📦 Move model to device (CPU or GPU)
    quantizer.model = quantizer.model.to(device)

    # 💾 Move captured input activations to same device
    quantizer.inps = quantizer.inps.to(device)

    # ⚙️ Move layer kwargs like attention_mask, position_ids, etc.
    quantizer.module_kwargs = {
        k: v.to(device) if isinstance(v, torch.Tensor) else v
        for k, v in quantizer.module_kwargs.items()
    }

    # 🔄 Move embeddings + rotary caches (cos/sin) to the same device
    quantizer.move_embed(quantizer.model, device)

    logger.info(f"🧠 Model, inps, kwargs, rotary now all on: {device}")
    logger.info("Model set up for activation extraction.")

    # === Step 3: Iterate over transformer blocks ===
    if quantizer.modules is None or quantizer.inps is None:
        raise RuntimeError("Calibration data (modules or inputs) not initialized.")

    for idx, module in enumerate(quantizer.modules):
        logger.info(
            f"\n🔍 [Block {idx}/{len(quantizer.modules)}] Getting input activations"
        )

        # ⚠️ Important: We assume module stays on CPU unless moved
        named_linears = get_named_linears(module)
        if not named_linears:
            logger.warning(f"No linear layers found in block {idx}, skipping")
            continue

        try:
            # 🎯 Forward pass to extract input activations
            #    This works only if:
            #    - model, inps, kwargs, rotary_emb => all on GPU or all on CPU
            input_feat = quantizer._get_input_feat(module, named_linears)
        except Exception as e:
            logger.exception(f"❌ Failed to get input for block {idx}: {e}")
            continue

        # === Step 4: Save the captured input features ===
        for name, act in input_feat.items():
            save_name = f"model.layers.{idx}.{name}"
            save_path = SAVE_DIR / f"{save_name}.pt"

            # 🧊 Always save to disk as CPU tensor for portability
            torch.save(act.cpu(), save_path)
            logger.info(f"✅ Saved activation: {save_path}, shape={tuple(act.shape)}")


if __name__ == "__main__":
    save_all_block_input_activations()
