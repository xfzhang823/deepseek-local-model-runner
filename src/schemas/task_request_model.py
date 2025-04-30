"""
schemas/task_request_model.py

This module defines the schema used to represent and validate LLM task requests.

It encapsulates:
- The type of task being requested (e.g., summarization, translation)
- The model backend to use (e.g., Hugging Face or AWQ)
- The generation mode (sampling strategy like precision or recall)
- An optional prompt type (only used by certain task types like keyword extraction)
- A flexible kwargs dictionary to carry task-specific fields (text, target_lang, etc.)

This model serves as the standardized contract between the frontend logic,
routing layer, and prompt/generation components.
"""

from pydantic import BaseModel, Field, model_validator
from typing import Optional, Literal, Dict, Any
from schemas.task_enums import TaskType, TaskMode, PromptType


class TaskRequestModel(BaseModel):
    """
    Pydantic model representing a single LLM task request.

    Fields:
        - task_type (TaskType): The type of LLM task (e.g., summarization,
        keyword extraction).
        - model (Literal["hf", "awq"]): The backend engine to use ("hf" for HuggingFace,
        - "awq" for quantized).
        - mode (TaskMode): Sampling behavior mode (e.g., precision, recall, balanced).
        - prompt_type (PromptType): Prompt formatting variant. Only applicable to
        certain tasks.
        - kwargs (Dict[str, Any]): Task-specific inputs such as text, target_lang, etc.

    Utility:
        - Enforces valid use of prompt_type
        - Supports task-specific input extraction (text, target_text, etc.)
    """

    task_type: TaskType
    model: Literal["hf", "awq"] = "hf"
    mode: TaskMode = TaskMode.BALANCED
    prompt_type: Optional[PromptType] = PromptType.DEFAULT

    kwargs: Dict[str, Any] = Field(default_factory=dict)

    def get_text_inputs(self) -> Dict[str, str]:
        """
        Extract key input fields relevant to most tasks.
        Useful for logging, prompting, and validation.

        Returns:
            dict: Dictionary of {field_name: value} for known task input fields.
        """
        keys = ["text", "source_text", "target_text", "target_lang"]
        return {k: v for k, v in self.kwargs.items() if k in keys}

    @model_validator(mode="before")
    @classmethod
    def validate_prompt_type_usage(cls, values):
        """
        Ensure `prompt_type` is only specified for supported task types.

        Currently, only TaskType.KEYWORD_EXTRACTION uses prompt variants meaningfully.

        Raises:
            ValueError: if prompt_type is set for an incompatible task.
        """
        task = values.get("task_type")
        prompt_type = values.get("prompt_type")

        if (
            task != TaskType.KEYWORD_EXTRACTION
            and prompt_type
            and prompt_type != PromptType.DEFAULT
        ):
            raise ValueError(
                f"`prompt_type` is only valid for {TaskType.KEYWORD_EXTRACTION}. Got: {task}"
            )
        return values
