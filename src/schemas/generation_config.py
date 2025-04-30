"""
project_config.py

Configurations for tweaking model temperature, top-k, and top-n.

* These are for DS local models only!

Sampling modes for keyword extraction:

1. Precision (Fewer, stricter keywords):
   - Goal: Maximize relevance, minimize noise.
   - Params: Low temp (0.3), high top_k (50), low top_p (0.7)
   - Use case: Clean outputs for filtering/analysis.

2. Balanced (Default):
   - Goal: Mix of relevance and coverage.
   - Params: Moderate temp (0.7), mid top_k (30), mid top_p (0.9)
   - Use case: General-purpose extraction.

3. Recall (More, broader keywords):
   - Goal: Maximize coverage, accept some noise.
   - Params: High temp (1.0), low top_k (10), high top_p (0.95)
   - Use case: Exploratory analysis or edge-case detection.

"""

from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict
from find_project_root import find_project_root
from schemas.task_enums import TaskMode, TaskType


# ——————————————————————————————————————————————————————
# 1) GENERATION CONFIG BASE
# ——————————————————————————————————————————————————————
@dataclass
class GenerationConfig:
    temperature: float
    top_k: int
    top_p: float
    max_new_tokens: int = 1500


# ——————————————————————————————————————————————————————
# 2) KEYWORD EXTRACTION
# ——————————————————————————————————————————————————————
@dataclass
class KeywordExtractionConfig(GenerationConfig):
    max_new_tokens: int = 2000


KEYWORD_EXTRACTION_PRESETS: Dict[TaskMode, KeywordExtractionConfig] = {
    TaskMode.PRECISION: KeywordExtractionConfig(temperature=0.45, top_k=30, top_p=0.75),
    TaskMode.BALANCED: KeywordExtractionConfig(temperature=0.5, top_k=40, top_p=0.85),
    TaskMode.RECALL: KeywordExtractionConfig(temperature=0.8, top_k=60, top_p=0.97),
    TaskMode.PURE_NUCLEUS: KeywordExtractionConfig(temperature=0.5, top_k=0, top_p=0.9),
}


# ——————————————————————————————————————————————————————
# 3) SUMMARIZATION
# ——————————————————————————————————————————————————————
SUMMARIZATION_PRESETS: Dict[TaskMode, GenerationConfig] = {
    TaskMode.PRECISION: GenerationConfig(
        temperature=0.5, top_k=60, top_p=0.8, max_new_tokens=300
    ),
    TaskMode.CREATIVE: GenerationConfig(
        temperature=1.2, top_k=15, top_p=1.0, max_new_tokens=500
    ),
}


# ——————————————————————————————————————————————————————
# 4) CONCEPT EXTRACTION
# ——————————————————————————————————————————————————————
CONCEPT_PRESETS: Dict[TaskMode, GenerationConfig] = {
    TaskMode.PRECISION: GenerationConfig(temperature=0.2, top_k=40, top_p=0.6),
    TaskMode.BALANCED: GenerationConfig(temperature=0.6, top_k=20, top_p=0.85),
}


# ——————————————————————————————————————————————————————
# 5) MASTER REGISTRY
# ——————————————————————————————————————————————————————
TASK_PRESETS: Dict[TaskType, Dict[TaskMode, GenerationConfig]] = {
    TaskType.KEYWORD_EXTRACTION: KEYWORD_EXTRACTION_PRESETS,
    TaskType.SUMMARIZATION: SUMMARIZATION_PRESETS,
    TaskType.CONCEPT_EXTRACTION: CONCEPT_PRESETS,
}


def get_config(task: TaskType, mode: TaskMode = TaskMode.BALANCED) -> GenerationConfig:
    """
    Fetch config for a task/mode combo, falling back to defaults if needed.

    Args:
        task (TaskType): The LLM task type (enum member).
        mode (TaskMode): Sampling preset for the task.

    Returns:
        GenerationConfig: The resolved configuration.

    Example:
        >>> from project_config import get_config, TaskType, TaskMode
        >>> cfg = get_config(TaskType.SUMMARIZATION, TaskMode.CREATIVE)
        >>> print(cfg)
        GenerationConfig(temperature=1.2, top_k=15, top_p=1.0, max_new_tokens=500)
    """
    presets = TASK_PRESETS.get(task.value if isinstance(task, TaskType) else task, {})
    # Return specific mode or default fallback
    return presets.get(
        mode if isinstance(mode, TaskMode) else TaskMode(mode),
        GenerationConfig(temperature=0.7, top_k=30, top_p=0.9),
    )
