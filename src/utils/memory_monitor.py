# utils/memory_monitor.py
import time
import psutil
import pynvml
import threading
import logging

logger = logging.getLogger(__name__)


def monitor_resources(interval=1):
    """Monitor CPU, RAM, and GPU VRAM usage at specified intervals."""
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)

    def log_stats():
        while True:
            cpu_percent = psutil.cpu_percent()
            ram_used = psutil.virtual_memory().used / (1024**3)
            gpu_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            gpu_used = gpu_info.used / (1024**2)
            gpu_total = gpu_info.total / (1024**2)

            logger.info(
                f"CPU: {cpu_percent}% | RAM: {ram_used:.2f} GB | GPU: {gpu_used:.2f}/{gpu_total:.2f} MB"
            )
            time.sleep(interval)

    thread = threading.Thread(target=log_stats, daemon=True)
    thread.start()


def start_monitoring(interval=1):
    """Start monitoring resources in a separate thread."""
    monitor_resources(interval=interval)
