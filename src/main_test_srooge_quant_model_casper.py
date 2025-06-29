import logging
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
model_path = Path(
    "~/models/casperhansen-deepseek-r1-distill-qwen-1.5b-awq"
).expanduser()

logger.info(f"model path: {model_path}")

# Sanitize before model loading
sanitize_config(
    config_path_or_dir=DEEPSEEK_R1_DISTILL_QUANT_MODEL_DIR,
    allowed_versions=["gemm", "gemv", "gemv_fast", "marlin"],
)

# Load model
model = AutoAWQForCausalLM.from_quantized(
    model_path,
    device="auto",  # ← ✅ Let accelerate manage placement
    # fuse_layers=False,  # optional, can improve speed
    safetensors=True,  # set to False if you used .bin
    torch_dtype=torch.float16,
    low_cpu_mem_usage=False,
)
df = summarize_quantization_structure(model)  # type: ignore[arg-type]
logger.info(df.head())

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
prompt = "What is the capital of France?"
inputs = tokenizer(prompt, return_tensors="pt").to(next(model.parameters()).device)


with torch.no_grad():
    out = model(**inputs)
    logits = out.logits
    print("NaNs?", torch.isnan(logits).any().item())
    print("Infs?", torch.isinf(logits).any().item())
    print("Min/max logits:", logits.min().item(), logits.max().item())

for name, mod in model.named_modules():
    if isinstance(mod, WQLinear_GEMM):
        print(f"\n🔍 Layer: {name}")
        print("  qweight:", getattr(mod, "qweight", None))
        print("  qzeros:", getattr(mod, "qzeros", None))
        print("  scales:", getattr(mod, "scales", None))
        if hasattr(mod, "qweight") and isinstance(mod.qweight, torch.Tensor):
            print("  qweight mean:", mod.qweight.float().abs().mean().item())
        break

# outputs = model.generate(
#     **inputs,
#     max_new_tokens=32,
#     do_sample=True,
#     temperature=0.8,
#     top_p=0.9,
#     top_k=50,
# )
# decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
# logger.info(f"🧠 Output: {decoded}")
