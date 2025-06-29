import logging
import os
from pathlib import Path
from awq import AutoAWQForCausalLM
from awq.modules.linear import WQLinear_GEMM
from transformers import AutoTokenizer
import torch
from project_config import DEEPSEEK_R1_DISTILL_QUANT_MODEL_DIR
from utils.sanitize_config import sanitize_config
from utils.audit_quant_model import (
    summarize_quantization_structure,
    audit_model_quantization,
)
import logging_config

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
)
# df = summarize_quantization_structure(model)  # type: ignore[arg-type]
# logger.info(df.head())

# audit_df = audit_model_quantization(model)  # type: ignore[arg-type]
# csv_file = (
#     Path("~/dev/deepseek_local_runner/documents").expanduser()
#     / "quant_audit_report.csv"
# )
# counts = audit_df["Module Type"].value_counts()
# logger.info(f"Module types:\n{counts}")

# # Save the report to disk
# audit_df.to_csv(csv_file, index=False)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)


for name, mod in model.named_modules():
    if isinstance(mod, WQLinear_GEMM):
        mod.register_forward_hook(check_nan_hook(name))

# Run a test prompt
prompt = "What is the capital of France?"
inputs = tokenizer(prompt, return_tensors="pt").to(next(model.parameters()).device)


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


for name, mod in model.named_modules():
    if isinstance(mod, WQLinear_GEMM):
        print(f"\n🔍 Layer: {name}")
        print("  qweight:", getattr(mod, "qweight", None))
        print("  qzeros:", getattr(mod, "qzeros", None))
        print("  scales:", getattr(mod, "scales", None))
        if hasattr(mod, "qweight") and isinstance(mod.qweight, torch.Tensor):
            print("  qweight mean:", mod.qweight.float().abs().mean().item())
        break

outputs = model.generate(
    **inputs,
    max_new_tokens=32,
    do_sample=True,
    temperature=0.8,
    top_p=0.9,
    top_k=50,
)
decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
logger.info(f"🧠 Output: {decoded}")
