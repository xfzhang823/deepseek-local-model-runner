"""
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

from dataclasses import dataclass
from typing import Literal

# ----------------------------
# Shared Types & Base Config
# ----------------------------
TaskMode = Literal["precision", "balanced", "recall", "exploratory"]


@dataclass
class GenerationConfig:
    temperature: float
    top_k: int
    top_p: float
    max_new_tokens: int = 1000  # Default for all tasks


# ----------------------------
# Task-Specific Presets
# ----------------------------


@dataclass
class KeywordExtractionConfig:
    """Sampling parameters for keyword extraction"""

    temperature: float
    top_k: int
    top_p: float
    max_new_tokens: int = 2000  # Shared default


# Distinct presets with clear tradeoffs
KEYWORD_EXTRACTION_PRESETS = {
    # Precision: Fewer, higher-confidence keywords
    # "precision": KeywordExtractionConfig(
    #     # temperature=0.3,  # Low randomness
    #     top_k=20,  # Focus on top tokens
    #     top_p=0.7,  # Narrow distribution
    # ),
    # todo: adjusted version; higher than default level to debug
    "precision": KeywordExtractionConfig(
        temperature=0.45,  # move a bit higher
        top_k=30,  # Focus on top tokens
        top_p=0.75,  # Narrow distribution
    ),
    # Balanced: Default middle ground
    "balanced": KeywordExtractionConfig(temperature=0.5, top_k=40, top_p=0.85),
    # Recall: More keywords (including edge cases)
    "recall": KeywordExtractionConfig(
        temperature=0.8,  # Higher diversity
        top_k=60,  # Wider token consideration
        top_p=0.97,  # Broad distribution
    ),
    "pure_nucleus": KeywordExtractionConfig(
        # temperature=0.6, #* setting 1
        temperature=0.5,  # * setting 2
        top_k=0,  # Disable top-k
        # top_p=0.90, #* setting 1
        top_p=0.9,  # * setting 2
    ),
}

CONCEPT_EXTRACTION_CONFIG = {
    "precision": GenerationConfig(
        temperature=0.2,  # Lower than keywords (concepts need stricter control)
        top_k=40,
        top_p=0.6,
    ),
    "balanced": GenerationConfig(temperature=0.6, top_k=20, top_p=0.85),
}

SUMMARIZATION_CONFIG = {
    "precision": GenerationConfig(
        temperature=0.5,  # Higher than extraction (needs some fluency)
        top_k=60,
        top_p=0.8,
        max_new_tokens=300,
    ),
    "creative": GenerationConfig(  # Different mode name for summarization
        temperature=1.2, top_k=15, top_p=1.0, max_new_tokens=500
    ),
}

# ----------------------------
# Master Config Registry
# ----------------------------
TASK_CONFIGS = {
    "keyword_extraction": KeywordExtractionConfig,
    "concept_extraction": CONCEPT_EXTRACTION_CONFIG,
    "summarization": SUMMARIZATION_CONFIG,
}


# ----------------------------
# Helper Functions
# ----------------------------
def get_config(task: str, mode: TaskMode = "balanced") -> GenerationConfig:
    """Fetch config for a task/mode combo with fallbacks."""
    configs = TASK_CONFIGS.get(task, {})
    return configs.get(
        mode, GenerationConfig(temperature=0.7, top_k=30, top_p=0.9)
    )  # Default fallback
