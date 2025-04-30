"""
src/artifacts.py

Utility for writing per-run YAML artifacts capturing inputs, configs, and outputs.

Each artifact is saved in the configured ARTIFACTS_DIR directory as `<timestamp>__<uuid8>.yaml`.
"""

import uuid
from datetime import datetime
from pathlib import Path
from dataclasses import asdict
import yaml

from project_config import ARTIFACTS_DIR

# Ensure the artifacts directory exists
ARTIFACT_DIR = Path(ARTIFACTS_DIR)
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)


def write_artifact(
    *,
    task_type: str,
    input_params: dict,
    config_obj,
    output_obj: dict,
) -> Path:
    """
    Write a YAML artifact for a single LLM run.

    Args:
        - task_type: Name of the LLM task (e.g., 'summarization').
        - input_params: Dict of input arguments passed to the task.
        - config_obj: Dataclass instance of GenerationConfig or similar.
        - output_obj: Dict representation of the Pydantic response (`.dict()`).

    Returns:
        Path to the written artifact file.
    """
    # Generate a unique, timestamped run ID
    timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%SZ")
    run_id = f"{timestamp}__{uuid.uuid4().hex[:8]}"
    filename = ARTIFACT_DIR / f"{run_id}.yaml"

    # Assemble payload: include settings and output
    payload = {
        "run_id": run_id,
        "task_type": task_type,
        "input": input_params,
        "settings": asdict(config_obj),
        "output": output_obj,
    }

    # Write YAML
    with open(filename, "w") as f:
        yaml.safe_dump(payload, f, sort_keys=False)

    return filename
