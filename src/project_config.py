from pathlib import Path
from find_project_root import find_project_root

# ——————————————————————————————————————————————————————
# ARTIFACTS DIR
# ——————————————————————————————————————————————————————

_raw_root = find_project_root()
if _raw_root is None:
    raise RuntimeError(
        "Could not find project root — `find_project_root()` returned None."
    )

root_dir = Path(_raw_root)

ARTIFACTS_DIR = root_dir / "artifacts"

# ——————————————————————————————————————————————————————
# Models DIR
# ——————————————————————————————————————————————————————
# DEEPSEEK_R1_DISTILL_FULL_MODEL_DIR = ""
DEEPSEEK_R1_DISTILL_QUANT_MODEL_OUTPUT_DIR = Path(
    "~/models/deepseek-r1-distill-qwen-1.5b-awq-scrooge-4bit-g128"
).expanduser()
DEEPSEEK_R1_DISTILL_QUANT_MODEL_DIR = (
    DEEPSEEK_R1_DISTILL_QUANT_MODEL_OUTPUT_DIR / "quantized_model"
)

DEEPSEEK_R1_DISTILL_HANSEN_QUANT_MODEL_DIR = Path(
    "~/models/casperhansen-deepseek-r1-distill-qwen-1.5b-awq"
).expanduser()
