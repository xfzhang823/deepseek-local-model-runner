"""dispatch_loader.py"""

from loaders.hf_loader import HF_ModelLoader
from loaders.awq_loader import AWQ_ModelLoader


def get_model_loader(loader_type: str = "hf"):
    """
    Dynamically select and load the model based on loader_type passed from caller.

    Args:
        loader_type (str): "hf" or "awq"

    Returns:
        (tokenizer, model)
    """
    loader_type = loader_type.lower()

    if loader_type == "awq":
        return AWQ_ModelLoader.load_model()
    elif loader_type == "hf":
        return HF_ModelLoader.load_model()
    else:
        raise ValueError(f"Unsupported MODEL_LOADER: {loader_type}")
