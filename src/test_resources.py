import logging
from utils.memory_monitor import start_monitoring
from utils.gpu_monitor import log_gpu_usage, clear_gpu_cache
from utils.inference_benchmark import measure_inference_speed
from utils.model_loader import load_model
import logging_config

resource_logger = logging.getLogger("resource")

# Start monitoring system resources
start_monitoring(interval=2)

# Load model with 8-bit quantization
tokenizer, model = load_model(
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", quantize="4bit"
)

# Check initial GPU usage
log_gpu_usage("Before Inference")

# Measure inference speed
input_text = "Explain the theory of relativity in simple terms."
elapsed, tokens_per_sec = measure_inference_speed(model, tokenizer, input_text)

# Log final GPU usage and clear cache
log_gpu_usage("After Inference")
clear_gpu_cache()
