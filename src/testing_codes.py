import time
from dotenv import load_dotenv
import os
import psutil
import pynvml
import torch
from threading import Thread


def monitor_resources(interval=1):
    """Monitor CPU, RAM, GPU VRAM in a separate thread."""
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # Use GPU 0

    while True:
        # CPU & RAM
        cpu_percent = psutil.cpu_percent()
        ram_used = psutil.virtual_memory().used / (1024**3)  # in GB

        # GPU VRAM
        gpu_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        gpu_used = gpu_info.used / (1024**2)  # in MB
        gpu_total = gpu_info.total / (1024**2)  # in MB

        print(
            f"CPU: {cpu_percent}% | "
            f"RAM: {ram_used:.2f} GB | "
            f"GPU VRAM: {gpu_used:.2f} MB / {gpu_total:.2f} MB"
        )
        time.sleep(interval)


def measure_inference_speed(model, tokenizer, input_text, max_new_tokens=50):
    """Benchmark LLM inference speed (tokens/sec)."""
    inputs = tokenizer(input_text, return_tensors="pt").to("cuda")
    start_time = time.time()

    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=max_new_tokens, pad_token_id=tokenizer.eos_token_id
        )

    elapsed = time.time() - start_time
    tokens_generated = len(outputs[0]) - len(inputs["input_ids"][0])
    tokens_per_sec = tokens_generated / elapsed

    print(
        f"Generated {tokens_generated} tokens in {elapsed:.2f}s ({tokens_per_sec:.2f} tokens/sec)"
    )


if __name__ == "__main__":
    # Start monitoring in background thread
    monitor_thread = Thread(target=monitor_resources, daemon=True)
    monitor_thread.start()

    # Load your LLM (example with Hugging Face)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    load_dotenv()
    model = os.getenv("MODEL_NAME_HF")
    print(model)

    tokenizer = AutoTokenizer.from_pretrained(model)
    model = AutoModelForCausalLM.from_pretrained(
        model,
        torch_dtype=torch.float16,
        device_map="auto",  # Auto offloads to CPU if GPU runs out of memory
    )

    # Test inference
    input_text = "Explain quantum computing in simple terms."
    measure_inference_speed(model, tokenizer, input_text)


import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import pynvml
from dotenv import load_dotenv
import os


def print_gpu_usage(step_name=""):
    """Print current GPU memory usage."""
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    used_mb = info.used / (1024**2)
    total_mb = info.total / (1024**2)
    print(f"{step_name}: GPU VRAM Used = {used_mb:.2f} MB / {total_mb:.2f} MB")
    pynvml.nvmlShutdown()


# 1. Check initial GPU usage (before loading)
print_gpu_usage("Before loading anything")

# 2. Load tokenizer (CPU-only, should not affect GPU)
load_dotenv()
model_name = os.getenv("MODEL_NAME_HF")
print(model_name)


tokenizer = AutoTokenizer.from_pretrained(model_name)
print_gpu_usage("After loading tokenizer")

# 3. Load model (this will consume GPU VRAM)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    # torch_dtype=torch.float8_e4m3fn,
    device_map="auto",  # Automatically offloads to CPU/GPU
)
print(f"Model dtype: {model.dtype}")  # Outputs torch.float16, torch.float32, etc.

print_gpu_usage("After loading model")

# Optional: Clear GPU cache to see baseline
torch.cuda.empty_cache()
print_gpu_usage("After emptying cache")
