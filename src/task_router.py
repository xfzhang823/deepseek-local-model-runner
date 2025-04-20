from typing import Any, Dict
from llm_response_models import LLMResponseBase
from task_manager import (
    summarize,
    translate,
    extract_keywords,
    generate_topics,
    align_texts,
)


def run_llm_task(task_type: str, **kwargs) -> LLMResponseBase:
    """
    Dispatch single LLM task based on type.
    """
    task_mapping = {
        "summarization": summarize,
        "translation": translate,
        "topic_generation": generate_topics,
        "text_alignment": align_texts,
        "keyword_extraction": extract_keywords,
    }

    if task_type not in task_mapping:
        return {"error": f"Unsupported task: {task_type}"}

    return task_mapping[task_type](**kwargs)
