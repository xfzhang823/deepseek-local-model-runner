"""
api_async.py

Async-first API layer for dispatching structured LLM task requests using `asyncio`.

This module is the entrypoint for:
- Single LLM task execution via `run_llm_task_async_api(request: TaskRequestModel)`
- Batch LLM task execution via
`run_llm_batch_task_async_api(requests: List[TaskRequestModel])`

All requests are schema-validated using Pydantic models, routed to
the appropriate task manager, and executed asynchronously using
asyncio-compatible wrappers.

-----------------------------------------------------
Architecture: DeepSeek Local - Async-Driven LLM Stack
-----------------------------------------------------

This module ties together all routing, config, and async wrappers through
structured API calls. It ensures separation of concerns between task logic,
prompt generation, and concurrency control.

Modules & Responsibilities

1. schemas/task_request_model.py
   - Defines `TaskRequestModel` (task_type, model, mode, prompt_type, kwargs).
   - Validates prompt usage, enforces enum choices, and normalizes input payload.

2. prompt_registry.py
   - Maps task_type and prompt_type to formatted prompt templates.
   - Validates required template fields and logs prompt construction behavior.

3. _sync_tasks.py
   - Implements all core task functions (summarize, translate,
   extract_keywords, etc.).
   - Each:
       a. Loads config via `get_config(task_type, mode)`
       b. Builds prompt using `get_prompt(...)`
       c. Calls `generate()` or `generate_with_thinking()`
       d. Parses tagged model outputs into structured Pydantic responses

4. task_manager_async.py
   - Wraps blocking sync task functions in `async def` wrappers using
   `run_in_executor(...)`
   - Provides non-blocking interfaces: `summarize_async(...)`, etc.

5. task_router_async.py
   - Validates dispatch options.
   - Maps each `TaskRequestModel` to the corresponding async handler with
   consistent argument passing.

6. batch_manager_async.py
   - Accepts a list of `TaskRequestModel`s.
   - Constructs async coroutine calls.
   - Executes via `asyncio.gather(...)` for concurrent batch task execution.

7. api_async.py (this module)
   - Top-level async API for local LLM interaction.
   - Offers:
       • `run_llm_task_async_api(request: TaskRequestModel)`
       • `run_llm_batch_task_async_api(requests: List[TaskRequestModel])`
   - Handles error resilience and structured return formatting.

* Call Flow (ASCII Diagram)

    Caller (async)                          Task Pipeline
    -------------                           -------------------------
    await run_llm_task_async_api(...)       +-------------------------+
       |                                    | task_router_async.py    |
       v                                    | run_llm_task_async(...) |
    +----------------------+                +-----------+-------------+
    |   src/api_async.py   |                            |
    |   run_llm_task_...   |                            v
    +----------------------+                +------------------------+
       |                                    | task_manager_async.py   |
       v                                    | summarize_async(...)    |
    +----------------------+                | translate_async(...)    |
    | task_router_async.py |                +-----------+-------------+
    | run_llm_task_async   |                            |
    +----------+-----------+                            v
               |                              +------------------------+
               v                              | llm_inference_async.py  |
    +----------------------+                  | generate / with_thinking |
    | _sync_tasks.py       | <--------------> | transformers.generate   |
    | blocking logic       |                  +------------------------+
    +----------------------+

Batching:
~~~~~~~~~

    +-----------------------------+
    | batch_manager_async.py      |
    | run_batch_tasks_async(...)  |
    +-------------+---------------+
                  |
    [ TaskRequestModel[] → coroutines ]
                  |
                  v
         await asyncio.gather(...)

Asynchronous Mechanics
~~~~~~~~~~~~~~~~~~~~~~

- **Single Event Loop**: All tasks run concurrently within one asyncio loop.
- **ThreadPoolExecutor**: All blocking generation (transformers) is offloaded via
`run_in_executor(...)`.
- **Strict Schema-Driven Input**: All calls use `TaskRequestModel` for clarity, safety,
and introspection.

Key Benefits
~~~~~~~~~~~~
- **Async-First Design**: High-throughput without blocking the main thread.
- **Schema-Validated**: Enforces clean separation of concerns between task config,
model backend, and payload.
- **Extensible**: New tasks, prompt types, or backends can be added with minimal coupling.
- **Unified Output**: All responses are returned as `.model_dump()` dictionaries,
ready for serialization.
"""

import asyncio
import logging
from typing import Any, Dict, List

from schemas.task_request_model import TaskRequestModel
from task_router_async import run_llm_task_async
from batch_manager_async import (
    run_batch_tasks_async,
    summarize_batch_async,
    translate_batch_async,
    keyword_extraction_batch_async,
    topic_generation_batch_async,
    text_alignment_batch_async,
)
import logging_config

logger = logging.getLogger(__name__)


# ——————————————————————————————————————————————————————
# ASYNC-ONLY API ENTRYPOINTS
# ——————————————————————————————————————————————————————
async def run_llm_task_async_api(request: TaskRequestModel) -> dict:
    """
    Asynchronous entrypoint for a single LLM task.

    Args:
        request (TaskRequestModel): Fully validated task request.

    Returns:
        dict: A JSON-serializable dict of the task result.
    """
    logger.info(
        f"[API] Starting single task: {request.task_type} | model={request.model}"
    )
    try:
        result = await run_llm_task_async(
            task_type=request.task_type,
            model=request.model,
            mode=request.mode,
            prompt_type=request.prompt_type,
            **request.kwargs,
        )
        logger.info(f"[API] Completed task: {request.task_type}")
        return result.model_dump()

    except Exception as e:
        logger.exception(f"[API] Task failed: {request.task_type}")
        return {
            "status": "error",
            "message": str(e),
            "task_type": request.task_type,
        }


async def run_llm_batch_task_async_api(
    requests: List[TaskRequestModel],
) -> List[dict]:
    """
    Asynchronous entrypoint for a batch of LLM tasks.

    Args:
        requests (List[TaskRequestModel]): Validated list of task requests.

    Returns:
        List[dict]: JSON-serializable list of responses.
    """
    logger.info(f"[API] Running batch task: {len(requests)} tasks")

    try:
        responses = await run_batch_tasks_async(requests)
        return [res.model_dump() for res in responses]

    except Exception as e:
        logger.exception("[API] Batch execution failed")
        return [
            {"status": "error", "message": str(e), "index": i}
            for i in range(len(requests))
        ]


# Convenience exports for testing/interactive use
__all__ = [
    "run_llm_task_async_api",
    "run_llm_batch_task_async_api",
    "summarize_batch_async",
    "translate_batch_async",
    "keyword_extraction_batch_async",
    "topic_generation_batch_async",
    "text_alignment_batch_async",
]
