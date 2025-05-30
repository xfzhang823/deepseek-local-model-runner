"""utils/offload_to_cpu.py"""

import logging
import gc
import torch

logger = logging.getLogger(__name__)


def offload_tensor_to_cpu(tensor: torch.Tensor) -> torch.Tensor:
    """
    Moves a tensor to CPU and releases the original GPU memory.

    Args:
        tensor (torch.Tensor): The original GPU tensor.

    Returns:
        torch.Tensor: CPU-resident version of the tensor.

    Raises:
        TypeError: If the input is not a torch.Tensor.
    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Expected a torch.Tensor, but got {type(tensor)} instead.")

    if not tensor.is_cuda:
        logger.info("Tensor is already on CPU.")
        return tensor

    cpu_tensor = tensor.cpu()
    del tensor
    gc.collect()
    torch.cuda.empty_cache()
    return cpu_tensor
