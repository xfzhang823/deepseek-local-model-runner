from pathlib import Path
from typing import Union, Optional
from dataclasses import dataclass, field
from utils.inspect_model import inspect_model, save_to_json, save_plots


@dataclass
class InspectConfig:
    layer_types: list[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "mlp"]
    )
    show_progress: bool = True


def analyze_and_save(
    model_name: str,
    output_dir: Union[Path, str] = "./inspection_results",
    config: Optional[InspectConfig] = None,
) -> None:
    """Run full inspection pipeline with error handling"""
    output_dir = Path(output_dir) if isinstance(output_dir, str) else output_dir
    try:
        config = config or InspectConfig()
        output_dir = Path(output_dir).expanduser()

        print("Starting model inspection..." if config.show_progress else "")
        results = inspect_model(model_name, config.layer_types)

        output_dir.mkdir(parents=True, exist_ok=True)
        save_to_json(results["stats"], output_dir / "stats.json")
        save_plots(results["plots"], output_dir / "plots")

        print(f"✅ Results saved to {output_dir.resolve()}")

    except Exception as e:
        print(f"❌ Analysis failed: {str(e)}")
        raise


if __name__ == "__main__":
    ds_r1_qwen_dist_model = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    save_to_dir = Path(
        "~/dev/deepseek_local_runner/documents/native_model_inspect"
    ).expanduser()
    analyze_and_save(model_name=ds_r1_qwen_dist_model, output_dir=save_to_dir)
