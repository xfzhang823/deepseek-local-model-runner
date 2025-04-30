from enum import Enum


class TaskType(str, Enum):
    """
    Enumeration of supported LLM task types.
    """

    CONCEPT_EXTRACTION = "concept_extraction"
    KEYWORD_EXTRACTION = "keyword_extraction"
    SUMMARIZATION = "summarization"
    TEXT_ALIGNMENT = "text_alignment"
    TOPIC_GENERATION = "topic_generation"
    TRANSLATION = "translation"

    @classmethod
    def has_value(cls, value: str) -> bool:
        return any(task.value == value for task in cls)

    def __str__(self) -> str:
        return self.value


class TaskMode(str, Enum):
    PRECISION = "precision"
    BALANCED = "balanced"
    RECALL = "recall"
    CREATIVE = "creative"
    PURE_NUCLEUS = "pure_nucleus"

    @classmethod
    def has_value(cls, value: str) -> bool:
        return any(mode.value == value for mode in cls)

    @classmethod
    def get_mode(cls, value: str):
        try:
            return cls(value)
        except ValueError:
            return None

    def __str__(self) -> str:
        return self.value


class PromptType(str, Enum):
    DEFAULT = "default"
    TECHNICAL = "technical"
    # BULLET = "bullet"
    # SHORT = "short"
    # Add more variants as needed
