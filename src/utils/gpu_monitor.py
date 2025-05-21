# utils/gpu_monitor.py
import logging
import pynvml
import torch
import logging_config

# Explicit resource logger
resource_logger = logging.getLogger("resource")


def log_gpu_usage(step_name=""):
    """Log GPU memory usage with a specific step label."""
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    info = pynvml.nvmlDeviceGetMemoryInfo(handle)

    used_mb = int(info.used) / (1024**2)
    total_mb = int(info.total) / (1024**2)

    resource_logger.info(
        f"{step_name}: GPU VRAM Used = {used_mb:.2f} MB / {total_mb:.2f} MB"
    )


def clear_gpu_cache():
    """Clear PyTorch GPU cache and log GPU usage."""

    torch.cuda.empty_cache()
    log_gpu_usage("After Clearing Cache")
