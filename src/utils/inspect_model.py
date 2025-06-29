from pathlib import Path
from typing import Dict, Any, List, Union
import json
import matplotlib.pyplot as plt
import torch
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
from accelerate import init_empty_weights
from tqdm import tqdm


def inspect_model(
    model_name: str,
    sample_text: str = "Hello world",
    layer_types: List[str] = ["q_proj", "k_proj", "mlp"],
    max_layers: int = 3,  # Start small for testing
) -> Dict[str, Union[Dict, plt.Figure]]:
    """Inspects weights AND activations with memory safety"""
    # Initialize results
    results = {"weights": {}, "activations": {}, "plots": {}}

    # 1. Load config and tokenizer
    config = AutoConfig.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    inputs = tokenizer(sample_text, return_tensors="pt")

    # 2. Layer-by-layer inspection
    with torch.no_grad():
        # First get all target layers
        with init_empty_weights():
            dummy_model = AutoModelForCausalLM.from_config(config)
            target_layers = [
                (name, param)
                for name, param in dummy_model.named_parameters()
                if any(lt in name for lt in layer_types)
            ][:max_layers]

        # Process each layer
        for name, _ in tqdm(target_layers, desc="Inspecting layers"):
            try:
                # --- WEIGHTS ---
                layer = AutoModelForCausalLM.from_pretrained(
                    model_name, torch_dtype=torch.float16, device_map={"": "cpu"}
                ).get_parameter(name)

                weights = layer.data.float().cpu()
                results["weights"][name] = {
                    "mean": weights.mean().item(),
                    "std": weights.std().item(),
                    "shape": list(weights.shape),
                }

                # --- ACTIVATIONS ---
                partial_model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16,
                    device_map={"": "cpu"},
                    output_attentions=True,
                ).eval()

                # Forward pass through partial model
                outputs = partial_model(**inputs.to("cpu"))

                # Record activation stats
                if hasattr(outputs, "attentions"):
                    results["activations"][f"{name}_attention"] = {
                        "mean": outputs.attentions[-1].mean().item(),
                        "max": outputs.attentions[-1].max().item(),
                    }

                # --- PLOTS ---
                fig, ax = plt.subplots(1, 2, figsize=(10, 4))
                ax[0].hist(weights.flatten().numpy(), bins=100)
                ax[0].set_title(f"{name} weights")

                if hasattr(outputs, "attentions"):
                    ax[1].imshow(outputs.attentions[-1][0, 0].cpu().numpy())
                    ax[1].set_title(f"{name} attention")

                results["plots"][name] = fig
                plt.close(fig)

                # Cleanup
                del layer, partial_model, outputs
                torch.cuda.empty_cache()

            except Exception as e:
                print(f"⚠️ Layer {name} failed: {str(e)}")
                continue

    return results


def calculate_stats(tensor: torch.Tensor) -> Dict[str, Union[float, List[int]]]:
    """Calculate layer statistics"""
    return {
        "mean": tensor.mean().item(),
        "std": tensor.std().item(),
        "min": tensor.min().item(),
        "max": tensor.max().item(),
        "shape": list(tensor.shape),
    }


def create_plot(tensor: torch.Tensor, layer_name: str):
    """Create histogram plot for a layer"""
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.hist(tensor.flatten().numpy(), bins=100)
    ax.set_title(f"{layer_name.split('.')[-1]} weights")
    plt.close(fig)  # Prevents inline display
    return fig


def save_to_json(data: Dict, path: Path) -> None:
    """Save inspection results to JSON"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def save_plots(plot_dict: Dict[str, plt.Figure], output_dir: Path) -> None:
    """Save all plots to PNG files"""
    Path(output_dir).mkdir(exist_ok=True)
    for name, fig in plot_dict.items():
        safe_name = name.replace(".", "_")
        fig.savefig(f"{output_dir}/{safe_name}.png")
        plt.close(fig)
