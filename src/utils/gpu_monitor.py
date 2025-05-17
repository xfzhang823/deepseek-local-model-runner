# utils/gpu_monitor.py
import pynvml
import logging

logger = logging.getLogger(__name__)


def log_gpu_usage(step_name=""):
    """Log GPU memory usage with a specific step label."""
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    used_mb = info.used / (1024**2)
    total_mb = info.total / (1024**2)
    logger.info(f"{step_name}: GPU VRAM Used = {used_mb:.2f} MB / {total_mb:.2f} MB")


def clear_gpu_cache():
    """Clear PyTorch GPU cache and log GPU usage."""
    import torch

    torch.cuda.empty_cache()
    log_gpu_usage("After Clearing Cache")
