from pathlib import Path
from find_project_root import find_project_root

# ——————————————————————————————————————————————————————
# ARTIFACTS DIR
# ——————————————————————————————————————————————————————
root_dir = Path(find_project_root())  # pyright: ignore[reportArgumentType]
ARTIFACTS_DIR = root_dir / "artifacts"

# ——————————————————————————————————————————————————————
# Models DIR
# ——————————————————————————————————————————————————————
DEEPSEEK_R1_DISTILL_FULL_MODEL_DIR = ""
DEEPSEEK_R1_DISTILL_QUANT_MODEL_OUTPUT_DIR = Path(
    "~/models/deepseek-r1-distill-qwen-1.5b-awq-scrooge-4bit-g128"
).expanduser()
DEEPSEEK_R1_DISTILL_QUANT_MODEL_DIR = (
    DEEPSEEK_R1_DISTILL_QUANT_MODEL_OUTPUT_DIR / "quantized_model"
)
