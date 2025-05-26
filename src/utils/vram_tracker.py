"""utils/vram_tracker.py"""

import threading
import time
from contextlib import contextmanager
import pynvml
import psutil
import logging

resource_logger = logging.getLogger("resource")


class VRAMMonitor:
    def __init__(self, interval=1):
        self.interval = interval
        self.thread = None
        self._stop_event = threading.Event()
        self.peak_gpu_used_mb = 0.0  # ✅ track peak gpu usage

    def _log_resources(self):
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)

        while not self._stop_event.is_set():
            cpu_percent = psutil.cpu_percent()
            ram_used = int(psutil.virtual_memory().used / (1024**3))
            gpu_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            gpu_used = int(gpu_info.used) / (1024**2)
            gpu_total = int(gpu_info.total) / (1024**2)

            self.peak_gpu_used_mb = max(self.peak_gpu_used_mb, gpu_used)

            resource_logger.info(
                f"[AutoMonitor] CPU: {cpu_percent:.1f}% | RAM: {ram_used:.2f} GB | "
                f"GPU: {gpu_used:.2f}/{gpu_total:.2f} MB"
            )
            time.sleep(self.interval)

    def start(self):
        self._stop_event.clear()
        self.thread = threading.Thread(target=self._log_resources, daemon=True)
        self.thread.start()

    def stop(self):
        self._stop_event.set()
        if self.thread is not None:
            self.thread.join()

    def get_peak_vram(self) -> float:
        return self.peak_gpu_used_mb


@contextmanager
def monitor_vram(interval=1):
    monitor = VRAMMonitor(interval=interval)
    monitor.start()
    try:
        yield monitor  # <-- Yield monitor so caller can access `.get_peak_vram()`
    finally:
        monitor.stop()
