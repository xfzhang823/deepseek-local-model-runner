import logging
import os
from pathlib import Path
from awq import AutoAWQForCausalLM
from awq.modules.linear import WQLinear_GEMM
from transformers import AutoTokenizer
import torch
from utils.sanitize_config import sanitize_config
from utils.audit_quant_model import (
    summarize_quantization_structure,
    audit_model_quantization,
    audit_quantized_layers_in_memory,
)
import logging_config
from project_config import DEEPSEEK_R1_DISTILL_QUANT_MODEL_DIR

logger = logging.getLogger(__name__)

# Path to your quantized model folder
model_path = DEEPSEEK_R1_DISTILL_QUANT_MODEL_DIR
logger.info(f"model path: {model_path}")

# Sanitize before model loading
sanitize_config(
    config_path_or_dir=DEEPSEEK_R1_DISTILL_QUANT_MODEL_DIR,
    allowed_versions=["gemm", "gemv", "gemv_fast", "marlin"],
)

os.environ["FLASH_ATTENTION_FORCE_DISABLE"] = "1"


def check_nan_hook(name):
    """
    Hook to log input and output tensor stats for debugging NaNs and Infs in forward passes.
    """

    def hook(module, input, output):
        input_tensor = (
            input[0] if isinstance(input, (tuple, list)) and len(input) > 0 else None
        )
        if input_tensor is not None:
            logger.debug(
                f"📥 {name} input: min={input_tensor.min().item():.4e}, max={input_tensor.max().item():.4e}"
            )
        if torch.isnan(output).any():
            logger.debug(f"❗ NaNs in output of {name}")
        if torch.isinf(output).any():
            logger.debug(f"❗ Infs in output of {name}")

    return hook


# Load model
model = AutoAWQForCausalLM.from_quantized(
    model_path,
    device="auto",  # ← ✅ Let accelerate manage placement
    fuse_layers=False,  # optional, can improve speed
    safetensors=True,  # set to False if you used .bin
    torch_dtype=torch.float16,
    low_cpu_mem_usage=False,
    trust_remote_code=True,
)


# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Register forward hook to check for NaN/Inf in all quantized layers
for name, mod in model.named_modules():
    if isinstance(mod, WQLinear_GEMM):
        mod.register_forward_hook(check_nan_hook(name))

# Run a test prompt
# prompt = "What is the capital of France?"
# prompt = "法国的首都是哪里？"
prompt = "What is 1 + 1?"

inputs = tokenizer(prompt, return_tensors="pt").to(next(model.parameters()).device)
input_ids = inputs["input_ids"]  # <--- This is the tensor

logger.info(f"Input IDs: {input_ids.tolist()}")
logger.info(f"Decoded input: {tokenizer.decode(input_ids[0])}")

# inputs: result of tokenizer(prompt, return_tensors="pt")
# model: your quantized model (e.g., model_q)
with torch.no_grad():
    out = model(**inputs)
    logits = out.logits

    has_nan = torch.isnan(logits).any().item()
    has_inf = torch.isinf(logits).any().item()
    min_val = logits.min().item()
    max_val = logits.max().item()

    logger.info(f"NaNs? {has_nan}")
    logger.info(f"Infs? {has_inf}")
    logger.info(f"Min/max logits: {min_val:.4e}, {max_val:.4e}")

# Inspect RoPE buffers
# Deep inside the model
transformer = model.model.model if hasattr(model.model, "model") else model.model
rope = transformer.rotary_emb

# Check if it has buffers (old style), otherwise say it's dynamic
has_cos = hasattr(rope, "cos_cached")
has_sin = hasattr(rope, "sin_cached")

if has_cos and has_sin:
    logger.info(
        f"RoPE device → cos: {rope.cos_cached.device}, sin: {rope.sin_cached.device}"
    )
else:
    logger.info("RoPE is dynamic (Qwen2-style) — cos/sin are generated at runtime.")


# Inspect attention
amask = inputs.get("attention_mask", None)
if amask is not None:
    logger.info(f"Attention mask → device: {amask.device}, shape: {tuple(amask.shape)}")
else:
    logger.warning("⚠️ Attention mask is missing from inputs!")


# For each quantized layer, log whether weight/qweight/qzeros/scales are present or missing
audit_quantized_layers_in_memory(model)

outputs = model.generate(
    **inputs,
    max_new_tokens=128,
    do_sample=False,
    temperature=0.6,
    top_p=0.9,
    top_k=50,
)
logger.info(f"Output IDs: {outputs[0].tolist()}")
logger.info(f"Decoded output: {tokenizer.decode(outputs[0])}")

decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
logger.info(f"🧠 Output: {decoded}")
