"""
prompts/prompt_registry.py

Central registry for resolving LLM prompts based on task type and prompt variant.

Responsibilities:
- Validates that all required input fields are present
- Selects the correct prompt template for a given task
- Dispatches dynamically to keyword prompt variants
- Logs prompt usage and helps track prompt-related issues

Expected Usage:
    from prompt_registry import get_prompt
    prompt = get_prompt(TaskType.SUMMARIZATION, text="...", ...)
"""

import logging
from schemas.task_enums import TaskType, PromptType
from .base_prompts import (
    SUMMARIZATION_PROMPT,
    TRANSLATION_PROMPT,
    TOPIC_GENERATION_PROMPT,
    TEXT_ALIGNMENT_PROMPT,
)
from prompts.keyword_extraction_prompts import get_keyword_prompt

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------------------
# Required formatting keys for each task
# ------------------------------------------------------------------------------

REQUIRED_PROMPT_FIELDS = {
    TaskType.SUMMARIZATION: ["text"],
    TaskType.TRANSLATION: ["text", "target_lang"],
    TaskType.TOPIC_GENERATION: ["text"],
    TaskType.TEXT_ALIGNMENT: ["source_text", "target_text"],
    TaskType.KEYWORD_EXTRACTION: ["text"],
}

# ------------------------------------------------------------------------------
# Prompt dispatcher
# ------------------------------------------------------------------------------


def get_prompt(
    task_type: TaskType,
    prompt_type: PromptType = PromptType.DEFAULT,
    **kwargs,
) -> str:
    """
    Returns a formatted prompt string based on the task type and optional prompt variant.

    Parameters:
        task_type (TaskType): The LLM task being performed (e.g., summarization).
        prompt_type (PromptType): Optional variant used for prompt selection.
        **kwargs: Task-specific fields such as 'text', 'source_text', 'target_lang'.

    Returns:
        str: Fully formatted prompt string to be passed into model generation.

    Raises:
        ValueError: If required input fields are missing or task type is not supported.
    """

    logger.debug(f"Resolving prompt for task: {task_type}, prompt_type: {prompt_type}")

    required = REQUIRED_PROMPT_FIELDS.get(task_type, [])
    missing = [field for field in required if field not in kwargs]
    if missing:
        msg = f"[{task_type}] Missing required fields for prompt: {missing}"
        logger.error(msg)
        raise ValueError(msg)

    try:
        # Dispatch per task
        if task_type == TaskType.SUMMARIZATION:
            prompt = SUMMARIZATION_PROMPT.format(**kwargs)

        elif task_type == TaskType.TRANSLATION:
            prompt = TRANSLATION_PROMPT.format(**kwargs)

        elif task_type == TaskType.TOPIC_GENERATION:
            prompt = TOPIC_GENERATION_PROMPT.format(**kwargs)

        elif task_type == TaskType.TEXT_ALIGNMENT:
            prompt = TEXT_ALIGNMENT_PROMPT.format(**kwargs)

        elif task_type == TaskType.KEYWORD_EXTRACTION:
            text = kwargs["text"]
            prompt = get_keyword_prompt(text, prompt_type)

        else:
            raise ValueError(f"Prompt not defined for task: {task_type}")

        logger.info(
            f"Prompt resolved for task: {task_type} | PromptType: {prompt_type}"
        )
        return prompt

    except KeyError as ke:
        msg = (
            f"Missing key '{ke.args[0]}' while formatting prompt for task: {task_type}"
        )
        logger.exception(msg)
        raise ValueError(msg) from ke

    except Exception as e:
        logger.exception(f"Failed to format prompt for task: {task_type}")
        raise
