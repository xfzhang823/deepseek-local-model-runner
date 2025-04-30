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

from schemas.generation_config import get_config, KEYWORD_EXTRACTION_PRESETS
from llm_inference import generate, generate_with_thinking
from llm_response_models import (
    SummarizationResponse,
    TranslationResponse,
    KeywordExtractionResponse,
    TopicGenerationResponse,
    TextAlignmentResponse,
)
from prompts.keyword_extraction_prompts import get_keyword_prompt
from prompts.base_prompts import (
    SUMMARIZATION_PROMPT,
    TRANSLATION_PROMPT,
    TOPIC_GENERATION_PROMPT,
    TEXT_ALIGNMENT_PROMPT,
)

logger = logging.getLogger(__name__)


# * Utils function for post inference parsing
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


# Task functions
def align_texts(
    source_text: str = None,
    target_text: str = None,
    mode: str = "balanced",
    prompt_type: str = "default",  # <-- even if unused
    model: str = "hf",
) -> TextAlignmentResponse:
    """
    Align key ideas or segments between a source and target text using an LLM.

    This task builds a prompt where the model is asked to compare two pieces of text
    and return the result inside an <alignment>...</alignment> tag. The response is
    extracted and returned as structured alignment content.

    Args:
        source_text (str): The original version of the text (required).
        target_text (str): The updated or compared version (required).
        mode (str): Sampling strategy (e.g., 'balanced').
        model (str): LLM backend identifier (e.g., 'hf', 'awq').

    Returns:
        TextAlignmentResponse: Parsed output from the <alignment> tag,
        with status, timing, and error fields.
    """
    logger.info(f"[SYNC] Running text alignment | model={model}, mode={mode}")

    if not source_text or not target_text:
        msg = "Missing required input(s): 'source_text' and/or 'target_text'"
        logger.error(f"[SYNC] Text alignment failed - {msg}")
        return TextAlignmentResponse(status="error", message=msg, alignment="")

    cfg = get_config("text_alignment", mode)
    prompt = TEXT_ALIGNMENT_PROMPT.format(
        source_text=source_text, target_text=target_text
    )
    logger.debug(f"[Prompt] {prompt}")

    try:
        thinking, answer = generate_with_thinking(prompt, cfg, model=model)
        alignment = extract_tagged_content(answer, "alignment")
        return TextAlignmentResponse(alignment=alignment, processing_time=thinking)
    except Exception as e:
        logger.exception(f"[SYNC] Text alignment failed: {e}")
        return TextAlignmentResponse(status="error", message=str(e), alignment="")


def extract_keywords(
    text: str = None,
    mode: str = "balanced",
    prompt_type: str = "default",
    model: str = "hf",
) -> KeywordExtractionResponse:
    """
    Extract keywords from the input text using the specified prompt type and model backend.

    The input is passed into a prompt formatted based on the `prompt_type`, and the LLM
    is expected to return keyword candidates inside a <keywords>...</keywords> tag.
    The result is parsed and returned as a structured list.

    Args:
        text (str): Input text to extract keywords from (required).
        mode (str): Sampling preset for keyword extraction (e.g., 'balanced').
        prompt_type (str): Variant of prompt to apply (e.g., 'default', 'technical').
        model (str): LLM backend identifier (e.g., 'hf', 'awq').

    Returns:
        KeywordExtractionResponse: A structured model containing extracted keyword list,
        processing metadata, and error status if applicable.
    """
    logger.info(
        f"[SYNC] Running keyword extraction | model={model}, mode={mode}, prompt_type={prompt_type}"
    )

    if not text or not text.strip():
        msg = "Missing required input: 'text'"
        logger.error(f"[SYNC] Keyword extraction failed - {msg}")
        return KeywordExtractionResponse(
            keywords=[], status="error", error=msg, processing_time_sec=0.0
        )

    cfg = KEYWORD_EXTRACTION_PRESETS.get(mode, KEYWORD_EXTRACTION_PRESETS["balanced"])
    prompt = get_keyword_prompt(text, prompt_type)
    logger.debug(f"[Prompt] {prompt}")

    start = time.time()
    try:
        output = generate(prompt, cfg, model=model)
        raw = output.get("response_text", "")
        kw_str = extract_tagged_content(raw, "keywords")
        keywords = [kw.strip() for kw in re.split(r",| ", kw_str) if kw.strip()]
        duration = time.time() - start
        return KeywordExtractionResponse(
            keywords=keywords, status="success", processing_time_sec=duration
        )
    except Exception as e:
        logger.exception(f"[SYNC] Keyword extraction failed: {e}")
        return KeywordExtractionResponse(
            keywords=[], status="error", error=str(e), processing_time_sec=0.0
        )


def generate_topics(
    text: str = None,
    mode: str = "balanced",
    prompt_type: str = "default",
    model: str = "hf",
) -> TopicGenerationResponse:
    """
    Generate high-level topics from the input text using a structured LLM prompt.

    The model is expected to return its output wrapped in a <topics>...</topics> tag,
    which is parsed and returned as a list of topic labels.

    Args:
        text (str): Input text to analyze (required, min 10 characters).
        mode (str): Sampling strategy for topic generation (e.g., 'balanced').
        model (str): LLM backend identifier (e.g., 'hf', 'awq').

    Returns:
        TopicGenerationResponse: A structured response containing a list of topics,
        processing time, and error metadata if applicable.
    """
    logger.info(f"[SYNC] Running topic generation | model={model}, mode={mode}")

    if not text or len(text.strip()) < 10:
        msg = "Input text too short (min 10 characters)"
        logger.error(f"[SYNC] Topic generation failed - {msg}")
        return TopicGenerationResponse(
            topics=[], status="error", message=msg, processing_time=""
        )

    cfg = get_config("topic_generation", mode)
    prompt = TOPIC_GENERATION_PROMPT.format(text=text)
    logger.debug(f"[Prompt] {prompt}")

    start = time.time()
    try:
        thinking, answer = generate_with_thinking(prompt, cfg, model=model)
        raw = extract_tagged_content(answer, "topics")
        topics = [t.strip() for t in re.split(r",|\n", raw) if t.strip()]
        duration = time.time() - start
        return TopicGenerationResponse(
            topics=topics, processing_time=f"{duration:.2f}s"
        )
    except Exception as e:
        logger.exception(f"[SYNC] Topic generation failed: {e}")
        return TopicGenerationResponse(
            topics=[], status="error", message=str(e), processing_time=""
        )


def summarize(
    text: str = None,
    mode: str = "balanced",
    prompt_type: str = "default",
    model: str = "hf",
) -> SummarizationResponse:
    """
    Generate a concise summary of the input text using the LLM.

    This function builds a prompt using a strict template that wraps the summary
    in a <summary>...</summary> tag. The model's output is parsed to extract
    the summary and returned in a structured response.

    Args:
        text (str): The input text to summarize (required).
        mode (str): Sampling preset for summarization (e.g., 'balanced').
        model (str): LLM backend identifier (e.g., 'hf', 'awq').

    Returns:
        SummarizationResponse: Parsed result from the <summary> tag,
        including processing metadata and error info if applicable.
    """
    logger.info(f"[SYNC] Running summarization | model={model}, mode={mode}")

    if not text or not text.strip():
        msg = "Missing required input: 'text'"
        logger.error(f"[SYNC] Summarization failed - {msg}")
        return SummarizationResponse(
            status="error", message=msg, summary="", processing_time=""
        )

    cfg = get_config("summarization", mode)
    prompt = SUMMARIZATION_PROMPT.format(text=text)
    logger.debug(f"[Prompt] {prompt}")

    try:
        output = generate(prompt, cfg, model=model)
        raw = output.get("response_text", "").replace("</think>", "").strip()
        summary = extract_tagged_content(raw, "summary").splitlines()[0].strip()

        return SummarizationResponse(
            summary=summary, processing_time=output.get("processing_time", "")
        )
    except Exception as e:
        logger.exception(f"[SYNC] Summarization failed: {e}")
        return SummarizationResponse(
            status="error", message=str(e), summary="", processing_time=""
        )


def translate(
    text: str = None,
    target_lang: str = None,
    mode: str = "balanced",
    prompt_type: str = "default",
    model: str = "hf",
) -> TranslationResponse:
    """
    Translate input text into a specified target language using an LLM.

    This function uses a strict prompt format where the model is instructed to
    return the result within a <translated>...</translated> tag. The response
    is parsed and returned in a structured output.

    Args:
        text (str): The text to be translated (required).
        target_lang (str): Target language for translation (required).
        mode (str): Sampling strategy for generation (e.g., 'balanced').
        model (str): LLM backend identifier (e.g., 'hf', 'awq').

    Returns:
        TranslationResponse: A structured response containing original and
        translated text, metadata, and optional error details.
    """
    logger.info(
        f"[SYNC] Running translation | model={model}, mode={mode}, target_lang={target_lang}"
    )

    if not text or not text.strip():
        msg = "Missing required input: 'text'"
        logger.error(f"[SYNC] Translation failed - {msg}")
        return TranslationResponse(
            original_text=text,
            translated_text="",
            target_language=target_lang or "",
            status="error",
            message=msg,
        )
    if not target_lang:
        msg = "Missing required input: 'target_lang'"
        logger.error(f"[SYNC] Translation failed - {msg}")
        return TranslationResponse(
            original_text=text,
            translated_text="",
            target_language="",
            status="error",
            message=msg,
        )

    cfg = get_config("translation", mode)
    prompt = TRANSLATION_PROMPT.format(text_to_translate=text, target_lang=target_lang)
    logger.debug(f"[Prompt] {prompt}")

    try:
        output = generate(prompt, cfg, model=model)
        translated = extract_tagged_content(
            output.get("response_text", ""), "translated"
        )
        return TranslationResponse(
            original_text=text,
            translated_text=translated,
            target_language=target_lang,
            processing_time=output.get("processing_time", ""),
        )
    except Exception as e:
        logger.exception(f"[SYNC] Translation failed: {e}")
        return TranslationResponse(
            original_text=text,
            translated_text="",
            target_language=target_lang,
            status="error",
            message=str(e),
        )
