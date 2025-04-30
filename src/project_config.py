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
DEEPSEEK_R1_DISTILL_QUANT_MODEL_SCROOGE_DIR = Path(
    "~/models/deepseek-awq-scrooge"
).expanduser()
