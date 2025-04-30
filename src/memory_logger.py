import torch
import logging

logger = logging.getLogger(__name__)


def cuda_memory_logger(tag: str = "Memory"):
    """
    Logs the current GPU memory usage (cuda memory).

    Args:
        tag (str, optional): Label to show in logs to describe context.
        Defaults to "Memory".

    Prints:
        Allocated, Reserved, and Max Reserved memory in MB.
    """
    if not torch.cuda.is_available():
        logger.info(f"🧠 [{tag}] CUDA not available. Skipping memory logging.")
        return

    allocated = torch.cuda.memory_allocated() / (1024**2)  # MB
    reserved = torch.cuda.memory_reserved() / (1024**2)  # MB
    max_reserved = torch.cuda.max_memory_reserved() / (1024**2)  # MB

    logger.info(
        f"🔍 [{tag}] Allocated: {allocated:.2f} MB, Reserved: {reserved:.2f} MB, Max Reserved: {max_reserved:.2f} MB"
    )
