"""
task_manager.py

Module that defines and validates task-specific functions for interacting with a
local LLM model. Tasks include:

- Summarization
- Translation
- Keyword Extraction
- Topic Generation
- Text Alignment

Each task sends a structured free-text prompt to the model, then extracts a response
from specific XML-style tags (e.g., <summary>...</summary>) returned by the model.
The output is validated against Pydantic response models.

Design Notes:
- Uses a singleton model loader to keep the model in memory.
- Follows a strict response format for extraction (tagged sections).
- Future-ready for JSON-formatted LLM responses (manual JSON parsing optional).
- Centralized logging for observability, validation, and debugging.

* Note: Use free text response and then manually creating JSON structure,
* instead of strict JSON format (optional for a later phase)
"""

import time
import re
import logging
from typing import List, Dict, Tuple
import torch
from pydantic import ValidationError

# User-defined modules
from model_loader import ModelLoader
from llm_response_models import (
    SummarizationResponse,
    TranslationResponse,
    KeywordExtractionResponse,
    TopicGenerationResponse,
    TextAlignmentResponse,
)
from key_word_extractor import KeywordExtractor
import logging_config

logger = logging.getLogger(__name__)


def generate_text(
    prompt: str,
    max_length: int = 1500,
    temperature: float = 0.7,
    top_k: int = 50,
    top_p: float = 0.9,
) -> Dict[str, str]:
    """
    Generate text from the model based on the given prompt and generation settings.

    This function handles tokenization, moves inputs to the correct device (GPU/CPU),
    performs generation, and returns the decoded text and generation duration.

    Args:
        - prompt (str): Input text to guide generation.
        - max_length (int): Maximum number of tokens in output.
        - temperature (float): Sampling temperature (0–2); higher is more random.
        ! default to 0.7 - like 0.3 in cloud LLM APIs
        - top_k (int): Top-k sampling threshold.
        - top_p (float): Top-p (nucleus) sampling threshold.

    Returns:
        Dict[str, str]: Dictionary with the model's output and generation time.
    """
    tokenizer, model = ModelLoader.load_model()
    inputs = tokenizer(prompt, return_tensors="pt")  # pylint: disable=not-callable
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    start_time = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            do_sample=True,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )

    elapsed = time.time() - start_time
    elapsed_time = time.strftime("%H:%M:%S", time.gmtime(elapsed))
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return {
        "response_text": output_text.strip(),
        "processing_time": f"{elapsed_time}",
    }


def generate_text_with_thinking(
    prompt: str,
    max_length: int = 2000,
    temperature: float = 0.7,
    top_k: int = 50,
    top_p: float = 0.9,
    thinking_tags: Tuple[str, str] = ("<think>", "</think>"),
    answer_tags: Tuple[str, str] = ("<answer>", "</answer>"),
) -> Tuple[str, str]:
    """
    Generates text with explicit thinking steps and final answer separation.

    Args:
        - prompt: Input prompt
        - max_length: Max tokens to generate
        - temperature: Sampling temperature (0-2)
        - top_k: Top-k sampling threshold
        - top_p: Top-p (nucleus) sampling threshold
        - thinking_tags: XML-style tags to wrap reasoning
        - answer_tags: XML-style tags to wrap final answer

    Returns:
        Tuple of (thinking_steps, final_answer)

    Example:
        thinking, answer = generate_text_with_thinking("Explain quantum computing")
    """
    # Load model and tokenizer
    tokenizer, model = ModelLoader.load_model()

    # Wrap prompt with thinking instructions
    start_think, end_think = thinking_tags
    start_answer, end_answer = answer_tags

    # wrapped_prompt = f"""
    # {start_think}
    # Analyze the request step-by-step before responding:
    # 1. Identify key concepts
    # 2. Plan response structure
    # 3. Verify factual consistency
    # {end_think}
    # {start_answer}
    # {prompt}
    # {end_answer}
    # """

    # todo: try this simple approach
    #     wrapped_prompt = f"""
    #     {prompt}
    #     <think>
    #     Please reason through the problem step by step without repeating yourself. \
    # Each step should be concise and progress logically toward the final answer.
    #     """
    wrapped_prompt = prompt

    logger.info(f"Wrapped prompt: {wrapped_prompt}")  # todo: debug; delete later

    # Generate raw output
    # pylint: disable=not-callable
    inputs = tokenizer(wrapped_prompt, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            do_sample=True,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            pad_token_id=tokenizer.eos_token_id,
        )
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

    logger.info(f"full output: \n{full_output}")  # todo: debug; delete later

    # Parse thinking and answer sections
    def extract_section(text, start_tag, end_tag):
        parts = text.split(start_tag)
        if len(parts) > 1:
            return parts[1].split(end_tag)[0].strip()
        return ""

    thinking_steps = extract_section(full_output, start_think, end_think)
    final_answer = extract_section(full_output, start_answer, end_answer)

    logger.debug(f"thinking steps: \n{thinking_steps}")  # todo: debug; delete later
    logger.debug(f"final answers: \n{final_answer}")  # todo: debug; delete later

    # Fallback if tags weren't respected
    if not final_answer:
        final_answer = full_output.replace(thinking_steps, "").strip()

    return thinking_steps, final_answer


def align_texts(source_text: str, target_text: str) -> TextAlignmentResponse:
    """
    Align corresponding ideas between a source and target text.

    Asks the model to return aligned pairs of key ideas or concepts inside an <alignment> tag.
    Useful for comparing revisions or tracking content transformation.

    Args:
        - source_text (str): The original version of the text.
        - target_text (str): The updated or compared version.

    Returns:
        TextAlignmentResponse: Pydantic model with aligned content and timing.
    """

    prompt = f"""
    Compare the following texts and align their main ideas.

    Return only the result in this format:
    <alignment>Aligned idea pairs or mappings</alignment>

    Source:
    {source_text}

    Target:
    {target_text}
    """
    try:
        output = generate_text(prompt)
        alignment = extract_tagged_content(output["response_text"], "alignment")
        validated = TextAlignmentResponse(
            alignment=alignment, processing_time=output["processing_time"]
        )
        logger.info("Text alignment succeeded")
        logger.debug(f"Full alignment LLM output:\n{output['response_text']}")
        return validated
    except ValidationError as ve:
        logger.warning(f"Text alignment validation failed: {ve}")
        return TextAlignmentResponse(
            status="error", message="Validation error", alignment=""
        )
    except Exception as e:
        logger.error(f"Text alignment error: {e}")
        return TextAlignmentResponse(status="error", message=str(e), alignment="")


def extract_keywords(
    text: str, mode: str = "precision", prompt_type: str = "default"
) -> KeywordExtractionResponse:
    """
    Optimized keyword extraction endpoint that:
    - Uses pre-configured modes (precision/balanced/recall)
    - Supports different prompt templates
    - Returns validated KeywordExtractionResponse

    Args:
        text: Input text to analyze
        mode: Extraction preset ('precision', 'balanced', 'recall',
        or 'pure_nucleus')
        prompt_type: Template key (e.g., 'technical', 'default')

    Returns:
        KeywordExtractionResponse with:
        - keywords: List[str] of extracted terms
        - status: 'success' or 'error'
        - error: Optional error message
        - processing_time_sec: Float duration
    """
    # Input validation
    if not text.strip():
        return KeywordExtractionResponse(
            keywords=[],
            status="error",
            error="Empty input text",
            processing_time_sec=0.0,
        )

    # Load model and execute extraction
    tokenizer, model = ModelLoader.load_model()
    extractor = KeywordExtractor(model=model, tokenizer=tokenizer)

    keywords_model = extractor.extract(
        text=text, mode=mode, prompt_type=prompt_type, max_keywords=3000
    )

    logger.info(f"Keywords extracted (model): {keywords_model}")
    return keywords_model


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
    # Find all possible matches
    matches = re.findall(rf"<{tag}>(.*?)</{tag}>", text, re.DOTALL)

    # Return the last match if found (most likely the actual summary)
    if matches:
        return matches[-1].strip()

    # Fallback: return the text after the last closing tag if no matches found
    last_close = max(text.rfind(f"</{tag}>"), text.rfind("</think>"))
    if last_close != -1:
        return text[last_close:].strip()

    return text.strip()


def generate_topics(text: str) -> TopicGenerationResponse:
    """
    Generate high-level topics from input text with strict formatting rules.

    Args:
        text: Input text to analyze (minimum 10 characters required)

    Returns:
        TopicGenerationResponse with:
        - topics: List of cleaned topics
        - processing_time: Formatted string
        - status/message for errors
    """
    # Input validation
    if not text or len(text.strip()) < 10:
        raise ValidationError("Input text too short (min 10 chars required)")

    prompt = f"""
    Extract key topics from this text as comma-separated values inside <topics> tags.
    STRICT RULES:
    1. Only include concepts DIRECTLY from this text: "{text[:500]}..."
    2. Never include examples/instructions
    3. Format EXACTLY: <topics>topic1, topic2</topics>
    """

    try:
        start_time = time.time()
        _, answer = generate_text_with_thinking(
            prompt,
            temperature=0.3,  # More deterministic
            max_length=800,  # Prevent verbose outputs
        )

        # Robust extraction
        topics_raw = extract_tagged_content(answer, "topics")
        topics = [
            re.sub(r"[<>]", "", t.strip())  # Remove any stray tags
            for t in re.split(r",|\n", topics_raw)
            if t.strip() and not t.lower().startswith(("example:", "return", "format"))
        ]

        return TopicGenerationResponse(
            topics=topics, processing_time=f"{time.time() - start_time:.2f}s"
        )

    except ValidationError as ve:
        logger.warning(f"Validation failed for text: '{text[:30]}...'")
        return TopicGenerationResponse(
            status="error", message=f"Invalid input: {str(ve)}", topics=[]
        )

    except Exception as e:
        logger.error(f"Failed processing text: '{text[:30]}...' | Error: {str(e)}")
        return TopicGenerationResponse(
            status="error", message="Processing error", topics=[]
        )


def summarize(text: str) -> SummarizationResponse:
    """
    Generate a concise summary of the input text using the LLM.

    Uses a clean, strict format to prompt the model to return a single
    <summary>...</summary> block. Result is validated and returned in a Pydantic model.

    Args:
        text (str): The input text to summarize.

    Returns:
        SummarizationResponse: Pydantic response model containing the summary and timing.
    """
    prompt = f"""
    Please summarize the following text in 1-3 sentences.
    
    STRICT INSTRUCTIONS:
    1. Your response must contain ONLY the summary
    2. Format EXACTLY like this: <summary>summary text here</summary>
    3. Do NOT include any other text, analysis, or tags
    
    Text to summarize:
    {text}
    """

    try:
        output = generate_text(prompt)
        logger.debug(f"Raw model output: {output}")  # Debugging

        # Clean the output
        cleaned_output = output["response_text"].replace("</think>", "").strip()
        summary = extract_tagged_content(cleaned_output, "summary")

        # Additional cleanup if needed
        summary = summary.split("\n")[0].strip()

        validated = SummarizationResponse(
            summary=summary, processing_time=output["processing_time"]
        )
        logger.info("Summarization succeeded")
        return validated

    except Exception as e:
        logger.error(f"Summarization error: {e}")
        return SummarizationResponse(
            status="error", message=str(e), summary="", processing_time="0 seconds"
        )


def translate(text: str, target_lang: str = "French") -> TranslationResponse:
    """
    Extract key terms from the input text using the LLM.

    The model is prompted to return a comma-separated list inside <keywords> tags.
    The result is parsed and validated, with optional limit on keyword count.

    Args:
        text (str): Text to extract keywords from.
        num_keywords (int): Maximum number of keywords to return (default: 5).

    Returns:
        KeywordExtractionResponse: Pydantic model containing extracted keywords
        and timing.
    """
    prompt = f"""
    **Translation Task**
    - Translate the following text to {target_lang}
    - Use natural, colloquial language
    - Respond ONLY with the translation in this exact format:
    <translated>TRANSLATION_HERE</translated>
    
    **Important Rules**
    1. DO NOT include the original text
    2. DO NOT add any explanations or notes
    3. DO NOT include any text outside the <translated> tags
    
    **Text to Translate**:
    "{text}"
    
    **Example**:
    For "Hello" to Spanish, you would respond:
    <translated>Hola</translated>
    """
    try:
        output = generate_text(prompt)
        translated_text = extract_tagged_content(output["response_text"], "translated")
        validated = TranslationResponse(
            original_text=text,
            translated_text=translated_text,
            target_language=target_lang,
            processing_time=output["processing_time"],
        )

        # ✅ Log SUCCESS (structured & readable)
        logger.info(
            f"Translation SUCCESS | "
            f"From: '{text[:30]}...' → To ({target_lang}): '{translated_text[:30]}...' | "
            f"Time: {output['processing_time']}"
        )

        # (Optional) Debug raw output if needed
        logger.debug(f"Full model output: {output['response_text']}")

        return validated
    except ValidationError as ve:
        logger.warning(f"Translation validation failed: {ve}")
        return TranslationResponse(
            status="error",
            message="Validation error",
            original_text=text,
            translated_text="",
            target_language=target_lang,
        )
    except Exception as e:
        logger.error(f"Translation error: {e}")
        return TranslationResponse(
            status="error",
            message=str(e),
            original_text=text,
            translated_text="",
            target_language=target_lang,
        )


# todo: old code; delete after debugging
# def extract_tagged_content(text: str, tag: str) -> str:
#     """
#     Extract content between custom XML-style tags.

#     Args:
#         text (str): The full LLM output.
#         tag (str): The tag to extract (e.g. 'topics', 'summary').

#     Returns:
#         str: The extracted content or the raw text if tag is not found.
#     """
#     match = re.search(rf"<{tag}>(.*?)</{tag}>", text, re.DOTALL)
#     return match.group(1).strip() if match else text.strip()

# def summarize(text: str) -> SummarizationResponse:
#     prompt = f"""
#     Please summarize the following text in 1-3 sentences.

#     Important instructions:
#     - Return ONLY the final summary
#     - Do NOT include any thinking process or analysis
#     - Format your response exactly like this: <summary>Your summary here.</summary>

#     Text:
#     {text}
#     """
#     try:
#         output = generate_text(prompt)

#         print(output)  # debugging

#         summary = extract_tagged_content(output["response_text"], "summary")
#         validated = SummarizationResponse(
#             summary=summary, processing_time=output["processing_time"]
#         )
#         logger.info("Summarization succeeded")
#         return validated
#     except ValidationError as ve:
#         logger.warning(f"Summarization validation failed: {ve}")
#         return SummarizationResponse(
#             status="error", message="Validation error", summary=""
#         )
#     except Exception as e:
#         logger.error(f"Summarization error: {e}")
#         return SummarizationResponse(status="error", message=str(e), summary="")


# def extract_topics(text: str) -> List[str]:
#     # Generate response with thinking steps
#     thinking_output, final_output = generate_text_with_thinking(
#         prompt=text,
#         max_length=1000,
#         temperature=0.3,  # Lower temp for less randomness
#     )

#     # Log thinking steps for debugging
#     logger.debug(f"Thinking Steps:\n{thinking_output}")

#     # Extract topics from the final answer
#     topics = []
#     if "<topics>" in final_output:
#         topics_str = final_output.split("<topics>")[1].split("</topics>")[0].strip()
#         topics = [t.strip() for t in topics_str.split(",")]

#     return topics


# def extract_keywords(text: str, max_keywords: int = 30) -> KeywordExtractionResponse:
#     """
#     Extract keywords without thinking steps, using a single constrained prompt.
#     The LLM determines optimal keyword quantity up to the specified maximum.
#     """
#     # Input validation
#     if not text.strip():
#         return KeywordExtractionResponse(
#             keywords=[],
#             status="error",
#             message="Empty input text",
#             processing_time="0s",
#         )

#     prompt = f"""
#     Extract the most relevant keywords from the following text.

#     Format your response like this:
#     <keywords>keyword1, keyword2, keyword3, ...</keywords>

#     Text:
#     {text}
#     """
#     # Character limit as fallback

#     try:
#         # Single generation call without thinking steps
#         output = generate_text(
#             prompt,
#             temperature=0.5,  # Balanced between deterministic and creative
#             max_length=300,  # Constrain output length
#             top_p=0.9,  # Focus on high-probability terms
#         )

#         # Robust extraction
#         raw_response = extract_tagged_content(output["response_text"], "keywords")

#         # Debug logging (enable only when needed)
#         logger.debug(f"Raw model response: {raw_response}")

#         # Robust extraction with multiple fallbacks
#         # Extraction with multiple fallbacks
#         if "<keywords>" in output["response_text"]:
#             keywords = extract_tagged_content(
#                 output["response_text"], "keywords"
#             ).split(",")
#         else:
#             keywords = output["response_text"].strip().split("\n")[0].split(",")

#         # Final cleaning
#         keywords = [
#             kw.strip()
#             for kw in keywords
#             if kw.strip() and not kw.lower().startswith(("example", "keyword"))
#         ][:max_keywords]

#         # keywords = [kw.strip() for kw in re.split(r",|\n", raw_response) if kw.strip()][
#         #     :max_keywords
#         # ]

#         if not keywords:
#             raise ValueError("No valid keywords found in response")

#         return KeywordExtractionResponse(
#             keywords=keywords,
#             processing_time=output["processing_time"],
#             status="success",
#         )

#     except Exception as e:
#         logger.error(f"Keyword extraction failed | " f"Error: {str(e)} | ")
#         return KeywordExtractionResponse(
#             keywords=[], status="error", message=f"Extraction failed: {str(e)}"
#         )


# def extract_keywords(text: str, max_keywords: int = 30) -> KeywordExtractionResponse:
#     """Your working version with bulletproof placeholder filtering"""
#     start_time = time.time()

#     # Validate input text
#     if not text.strip():
#         return KeywordExtractionResponse(
#             keywords=[],
#             status="error",
#             message="Empty input text",
#             processing_time="0s",
#         )

#     # Prompt with thinking and strict answer format
#     #     prompt = f"""
#     #     Extract the most relevant keywords from the following text.

#     #     INSTRUCTIONS:
#     #     1. In <think>...</think>, analyze the text step-by-step to identify key terms.
#     #     2. In <answer>...</answer>, output ONLY: <keywords>word1, word2, word3</keywords>
#     #     3. NO extra text outside or inside the <answer> tags
#     #     4. NO placeholders (e.g., 'keyword1', 'example')
#     #     5. Use only words/phrases directly from the text
#     #     6. Separate keywords with commas and a space
#     #     7. Limit to {max_keywords} keywords max

#     #     EXAMPLE:
#     #     Text: "Cats chase mice in the house."
#     #     Output: <think>Cats are subjects, chase is action, mice and house are key nouns.</think>\
#     # <answer><keywords>cats, chase, mice, house</keywords></answer>

#     #     Text:
#     #     {text[:2000]}
#     #     """
#     prompt = f"""
#     Extract the most relevant keywords from this text: "{text}"

#     <think>
#     1. Identify ONLY the core technical terms (ignore descriptions, examples, verbs).
#     2. Keywords must be:
#     - Nouns or noun phrases (e.g., "neural networks," not "excel at pattern recognition").
#     - Directly mentioned in the text (no paraphrasing).
#     3. Output MUST follow this format:
#     [KEYWORDS]keyword1, keyword2, keyword3[/KEYWORDS]
#     - No extra text, explanations, or original sentences.
#     </think>
#     """
#     try:
#         # Use generate_text_with_thinking with tighter max_length
#         thinking_steps, final_answer = generate_text_with_thinking(
#             prompt=prompt,
#             temperature=0.5,
#             max_length=2000,
#             top_k=50,
#             top_p=0.9,  # Default working params
#             thinking_tags=("<think>", "</think>"),
#             answer_tags=("<answer>", "</answer>"),
#         )
#         logger.info(f"Thinking steps: {thinking_steps}")
#         logger.info(f"Raw answer: {final_answer}")

#         # Extract keywords from the answer
#         keywords_str = extract_tagged_content(final_answer, "keywords")
#         if not keywords_str:
#             # Fallback if tags are missing
#             keywords_str = re.sub(r"<[^>]+>", "", final_answer).strip()

#         # Split, clean, and filter (out typical wrong) keywords
#         keywords = [
#             kw.strip()
#             for kw in keywords_str.split(",")
#             if kw.strip()
#             and kw.strip() not in {"keyword1", "keyword2", "example", "kw1", "kw2"}
#         ][:max_keywords]

#         # Fallback if no valid keywords
#         if not keywords:
#             keywords = ["general_topic"]
#             logger.warning("No valid keywords extracted; using fallback.")

#         # Calculate processing time (since generate_text_with_thinking doesn't return it)
#         elapsed = (
#             time.time() - start_time
#         )  # Placeholder; you'd need to time it explicitly
#         elapsed_time = time.strftime("%H:%M:%S", time.gmtime(elapsed))

#         # Success logging
#         logger.info(
#             f"Keyword extraction succeeded | Found {len(keywords)} keywords | "
#             f"Sample: {keywords[:5]} | "
#             f"Time: {elapsed_time}"
#         )

#         validated = KeywordExtractionResponse(
#             keywords=keywords,
#             processing_time=elapsed_time,
#             status="success",
#         )

#     except Exception as e:
#         logger.error(f"Keyword extraction failed: {str(e)}")
#         return KeywordExtractionResponse(
#             keywords=["general_topic"],
#             status="error",
#             message=str(e),
#             processing_time="0s",
#         )
