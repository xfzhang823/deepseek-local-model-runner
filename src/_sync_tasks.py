"""
_sync_tasks.py

Synchronous task implementations that wrap centralized LLM inference
and allow selecting between HF, AWQ, or other LLM providers via a `model` parameter.

Tasks include:
- Summarization
- Translation
- Keyword Extraction
- Topic Generation
- Text Alignment

Each task:
1. Fetches a `GenerationConfig` via `get_config(task_name, mode)`
2. Builds a prompt from `prompts.py` or `prompts_keyword_extraction.py`
3. Calls `generate(...)` or `generate_with_thinking(...)` from `llm_inference`
4. Parses the output via `extract_tagged_content`
5. Returns a Pydantic response model

Supported models:
- "hf"  → Hugging Face 4/8-bit models
- "awq" → AWQ-quantized models

Design Notes:
- Uses centralized loaders under the hood
- Follows a strict response format with tagged sections
- Future-ready for JSON outputs with minimal parsing changes
- Centralized logging for observability and debugging
"""
import time
import re
import logging
from typing import List
from pydantic import ValidationError

from project_config import get_config, KEYWORD_EXTRACTION_PRESETS
from llm_inference import generate, generate_with_thinking
from llm_response_models import (
    SummarizationResponse,
    TranslationResponse,
    KeywordExtractionResponse,
    TopicGenerationResponse,
    TextAlignmentResponse,
)
from prompts_keyword_extraction import get_keyword_prompt
from prompts import (
    SUMMARIZATION_PROMPT,
    TRANSLATION_PROMPT,
    TOPIC_GENERATION_PROMPT,
    TEXT_ALIGNMENT_PROMPT,
)

logger = logging.getLogger(__name__)


def extract_tagged_content(text: str, tag: str) -> str:
    """
    Extract content from a specific XML-style tag in the model's output.

    Handles noisy or malformed output and returns either the latest matched tag content
    or a fallback segment of the raw text.

    Args:
        text (str): The full text output from the model.
        tag (str): The tag name to extract from (e.g., "summary", "topics").

    Returns:
        str: The extracted content or raw fallback string if no tag is found.
    """
    matches = re.findall(rf"<{tag}>(.*?)</{tag}>", text, re.DOTALL)
    if matches:
        return matches[-1].strip()
    last_close = text.rfind(f"</{tag}>")
    return text[last_close:].strip() if last_close != -1 else text.strip()


def align_texts(
    source_text: str,
    target_text: str,
    mode: str = "balanced",
    model: str = "hf"
) -> TextAlignmentResponse:
    """
    Align corresponding ideas between a source and target text.

    Asks the model to return aligned pairs of key ideas or concepts inside an
    <alignment> tag. Useful for comparing revisions or tracking content
    transformation.

    Args:
        - source_text (str): The original version of the text.
        - target_text (str): The updated or compared version.

    Returns:
        TextAlignmentResponse: Pydantic model with aligned content and timing.
    """
    cfg = get_config("text_alignment", mode)
    prompt = TEXT_ALIGNMENT_PROMPT.format(
        source_text=source_text,
        target_text=target_text
    )
    try:
        thinking, answer = generate_with_thinking(prompt, cfg, backend=model)
        alignment = extract_tagged_content(answer, "alignment")
        return TextAlignmentResponse(
            alignment=alignment,
            processing_time=thinking
        )
    except ValidationError as ve:
        logger.warning(f"Alignment validation failed: {ve}")
        return TextAlignmentResponse(status="error", message="Validation error", alignment="")
    except Exception as e:
        logger.error(f"Alignment error: {e}")
        return TextAlignmentResponse(status="error", message=str(e), alignment="")


def extract_keywords(
    text: str,
    mode: str = "balanced",
    prompt_type: str = "default",
    model: str = "hf"
) -> KeywordExtractionResponse:
    """
    Extract keywords using pre-configured sampling modes and prompt templates.

    Args:
        text (str): Input text to analyze. Must not be empty.
        mode (str): Sampling preset for extraction ('precision', 'balanced', 'recall').
        prompt_type (str): Template key for prompting (e.g., 'default', 'technical').
        model (str): LLM backend identifier (e.g., 'hf', 'awq').

    Returns:
        KeywordExtractionResponse: Pydantic model containing:
            - keywords: List[str] of extracted keywords
            - status: 'success' or 'error'
            - error: Optional[str] error message
            - processing_time_sec: float processing duration
    """
    if not text.strip():
        return KeywordExtractionResponse(
            keywords=[], status="error", error="Empty input text", processing_time_sec=0.0
        )

    cfg = KEYWORD_EXTRACTION_PRESETS.get(mode, KEYWORD_EXTRACTION_PRESETS["balanced"])
    prompt = get_keyword_prompt(text, prompt_type)
    start = time.time()
    output = generate(prompt, cfg, backend=model)
    raw = output.get("response_text", "")
    kw_str = extract_tagged_content(raw, "keywords")
    keywords = [kw.strip() for kw in re.split(r",| ", kw_str) if kw.strip()]
    duration = time.time() - start
    return KeywordExtractionResponse(
        keywords=keywords,
        status="success",
        processing_time_sec=duration
    )(
            keywords=[], status="error", error="Empty input text", processing_time_sec=0.0
        )

def generate_topics(
    text: str,
    mode: str = "balanced",
    model: str = "hf"
) -> TopicGenerationResponse:
    """
    Generate high-level topics from input text with strict formatting rules.

    Args:
        text: Input text to analyze (minimum 10 characters required)

    Returns:
        TopicGenerationResponse with:
        - topics: List of cleaned topics
        - processing_time: Formatted string
        - status/message for errors.
    """
    if not text or len(text.strip()) < 10:
        raise ValidationError("Input text too short (min 10 chars)")

    cfg = get_config("topic_generation", mode)
    prompt = TOPIC_GENERATION_PROMPT.format(text=text)
    start = time.time()
    thinking, answer = generate_with_thinking(prompt, cfg, backend=model)
    raw = extract_tagged_content(answer, "topics")
    topics = [t.strip() for t in re.split(r",|\n", raw) if t.strip()]
    duration = time.time() - start
    return TopicGenerationResponse(
        topics=topics,
        processing_time=f"{duration:.2f}s"
    )


def summarize(
    text: str,
    mode: str = "balanced",
    model: str = "hf"
) -> SummarizationResponse:
    """
    Generate a concise summary of the input text using the LLM.

    Uses a clean, strict format to prompt the model to return a single
    <summary>...</summary> block. Result is validated and returned in a Pydantic model.

    Args:
        text (str): The input text to summarize.

    Returns:
        SummarizationResponse: Pydantic response model containing the summary and timing.
    """
    cfg = get_config("summarization", mode)
    prompt = SUMMARIZATION_PROMPT.format(text=text)
    try:
        output = generate(prompt, cfg, backend=model)
        raw = output.get("response_text", "").replace("</think>", "").strip()
        summary = extract_tagged_content(raw, "summary").splitlines()[0].strip()
        return SummarizationResponse(
            summary=summary,
            processing_time=output.get("processing_time", "")
        )
    except Exception as e:
        logger.error(f"Summarization error: {e}")
        return SummarizationResponse(status="error", message=str(e), summary="", processing_time="")


def translate(
    text: str,
    target_lang: str = "French",
    mode: str = "balanced",
    model: str = "hf"
) -> TranslationResponse:
    """
    Translate input text into a target language using strict <translated> tags.

    Args:
        text (str): The input text to translate.
        target_lang (str): The language to translate into (e.g., 'French', 'German').
        mode (str): Sampling preset for translation ('precision', 'balanced', 'recall').
        model (str): LLM backend identifier (e.g., 'hf', 'awq').

    Returns:
        TranslationResponse: Pydantic model containing:
            - original_text: str
            - translated_text: str
            - target_language: str
            - processing_time: str duration
            - status: 'success' or 'error'
            - message: Optional error message
    """
    cfg = get_config("translation", mode)
    prompt = TRANSLATION_PROMPT.format(
        text_to_translate=text,
        target_lang=target_lang
    )
    try:
        output = generate(prompt, cfg, backend=model)
        translated = extract_tagged_content(output.get("response_text", ""), "translated")
        return TranslationResponse(
            original_text=text,
            translated_text=translated,
            target_language=target_lang,
            processing_time=output.get("processing_time", "")
        )
    except ValidationError as ve:
        logger.warning(f"Translation validation failed: {ve}")
        return TranslationResponse(status="error", message="Validation error", original_text=text, translated_text="", target_language=target_lang)
    except Exception as e:
        logger.error(f"Translation error: {e}")
        return TranslationResponse(status="error", message=str(e), original_text=text, translated_text="", target_language=target_lang)
(
            original_text=text,
            translated_text=translated,
            target_language=target_lang,
            processing_time=output.get("processing_time", "")
        )
    except ValidationError as ve:
        logger.warning(f"Translation validation failed: {ve}")
        return TranslationResponse(status="error", message="Validation error", original_text=text, translated_text="", target_language=target_lang)
    except Exception as e:
        logger.error(f"Translation error: {e}")
        return TranslationResponse(status="error", message=str(e), original_text=text, translated_text="", target_language=target_lang)
