"""
TODO: May not need this module - delete later!

llm_inference_async.py


Async wrappers for blocking LLM inference functions.
* Passes sync functions to a thread‑pool executor to run in async pipelines.

This module provides:

- `llm_call_async(...)`: Asynchronously runs `generate` in a thread-pool executor.
- `llm_call_with_thinking_async(...)`: Asynchronously runs `generate_with_thinking`
in a thread-pool executor.

By offloading the heavy `.generate()` calls to background threads via
`asyncio.get_running_loop().run_in_executor`, these functions let you
integrate LLM inference into `asyncio` workflows without blocking the event loop.
"""

import asyncio
from src._sync_tasks import generate, generate_with_thinking


async def llm_call_async(*args, **kwargs):
    """
    Asynchronously execute the blocking `generate` function on a background thread.
    """
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, generate, *args, **kwargs)


async def llm_call_with_thinking_async(*args, **kwargs):
    """
    Asynchronously execute the blocking `generate_with_thinking` function on
    a background thread.
    """
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, generate_with_thinking, *args, **kwargs)
