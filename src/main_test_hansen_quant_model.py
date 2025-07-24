import logging
from pathlib import Path
from awq import AutoAWQForCausalLM
from awq.modules.linear import WQLinear_GEMM
from transformers import AutoTokenizer
import torch
from project_config import DEEPSEEK_R1_DISTILL_HANSEN_QUANT_MODEL_DIR
from utils.sanitize_config import sanitize_config
from utils.audit_quant_model import (
    summarize_quantization_structure,
    audit_model_quantization,
)
import logging_config

logger = logging.getLogger(__name__)

# Path to your quantized model folder
model_path = DEEPSEEK_R1_DISTILL_HANSEN_QUANT_MODEL_DIR

logger.info(f"Model path: {model_path}")

# Sanitize before model loading
sanitize_config(
    config_path_or_dir=DEEPSEEK_R1_DISTILL_HANSEN_QUANT_MODEL_DIR,
    allowed_versions=["gemm", "gemv", "gemv_fast", "marlin"],
)

# Load model
model = AutoAWQForCausalLM.from_quantized(
    model_path,
    device="auto",  # ← ✅ Let accelerate manage placement
    fuse_layers=False,  # optional, can improve speed
    safetensors=True,  # set to False if you used .bin
    torch_dtype=torch.float16,
    low_cpu_mem_usage=False,
)
df = summarize_quantization_structure(model)  # type: ignore[arg-type]
logger.info(f"\n{df.head()}")

audit_df = audit_model_quantization(model)  # type: ignore[arg-type]
csv_file = (
    Path("~/dev/deepseek_local_runner/documents").expanduser()
    / "quant_audit_report_casper_model.csv"
)
counts = audit_df["Module Type"].value_counts()
logger.info(f"Module types:\n{counts}")

# Save the report to disk
audit_df.to_csv(csv_file, index=False)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Run a test prompt
# prompt = "What is the capital of France?"
prompt = "What is 1 + 1?"
inputs = tokenizer(prompt, return_tensors="pt").to(next(model.parameters()).device)

with torch.no_grad():
    out = model(**inputs)
    logits = out.logits
    logger.info(f"NaNs? {torch.isnan(logits).any().item()}")
    logger.info(f"Infs? {torch.isinf(logits).any().item()}")
    logger.info(f"Min/max logits: {logits.min().item()} / {logits.max().item()}")

for name, mod in model.named_modules():
    if isinstance(mod, WQLinear_GEMM):
        logger.info(f"\n🔍 Layer: {name}")
        logger.info(f"  qweight: {getattr(mod, 'qweight', None)}")
        logger.info(f"  qzeros: {getattr(mod, 'qzeros', None)}")
        logger.info(f"  scales: {getattr(mod, 'scales', None)}")
        if hasattr(mod, "qweight") and isinstance(mod.qweight, torch.Tensor):
            logger.info(f"  qweight mean: {mod.qweight.float().abs().mean().item()}")
        break

outputs = model.generate(
    **inputs,
    max_new_tokens=128,  # Just testing with shorter prompt
    do_sample=True,
    temperature=0.4,  # Set this low for testing (0 - 2.0 range)
    top_p=0.9,
    top_k=50,
)
decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
logger.info(f"🧠 Output: {decoded}")
