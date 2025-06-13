import logging
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer
import torch
from project_config import DEEPSEEK_R1_DISTILL_QUANT_MODEL_DIR
from utils.sanitize_config import sanitize_config
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

# Load model
model = AutoAWQForCausalLM.from_quantized(
    model_path,
    device="auto",  # ← ✅ Let accelerate manage placement
    fuse_layers=False,  # optional, can improve speed
    safetensors=True,  # set to False if you used .bin
    torch_dtype=torch.float16,
    low_cpu_mem_usage=False,
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Run a test prompt
inputs = tokenizer("What is the capital of France?", return_tensors="pt").to(
    model.device
)
outputs = model.generate(**inputs, max_new_tokens=32)

# Decode and print
logger.info(tokenizer.decode(outputs[0], skip_special_tokens=True))
