"""
task_router_async.py

Central dispatcher for LLM task execution.

Receives a validated `TaskRequestModel` and routes it to the appropriate
asynchronous task handler (e.g., summarization, translation, etc.).

Each task handler wraps its sync implementation using `run_in_executor`,
and returns a structured Pydantic response model.
"""

import logging
from typing import Callable, Awaitable
from schemas.task_request_model import TaskRequestModel
from schemas.task_enums import TaskType
from llm_response_models import LLMResponseBase
from task_manager_async import (
    summarize_async,
    translate_async,
    extract_keywords_async,
    generate_topics_async,
    align_texts_async,
)

logger = logging.getLogger(__name__)

# Map of task handlers
TASK_HANDLER_MAP: dict[TaskType, Callable[..., Awaitable[LLMResponseBase]]] = {
    TaskType.SUMMARIZATION: summarize_async,
    TaskType.TRANSLATION: translate_async,
    TaskType.KEYWORD_EXTRACTION: extract_keywords_async,
    TaskType.TOPIC_GENERATION: generate_topics_async,
    TaskType.TEXT_ALIGNMENT: align_texts_async,
}


async def run_llm_task_async(request: TaskRequestModel) -> LLMResponseBase:
    """
    Route a validated task request to the appropriate handler and return
    its result.

    Args:
        request (TaskRequestModel): Validated task request model.

    Returns:
        LLMResponseBase: The structured result of the LLM task.
    """
    logger.info(f"[Router] Dispatching task: {request.task_type}")

    handler = TASK_HANDLER_MAP.get(request.task_type)

    if handler is None:
        raise ValueError(f"No handler defined for task: {request.task_type}")

    try:
        return await handler(
            model=request.model,
            mode=request.mode,
            prompt_type=request.prompt_type,
            **request.kwargs,
        )
    except Exception as e:
        logger.exception(f"[Router] Task failed: {request.task_type}")
        raise
