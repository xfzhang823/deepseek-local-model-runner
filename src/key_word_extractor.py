"""Extractor class for keywords (including local LLM generation)"""

import re
import ast
import logging
import time
from typing import List, Optional
import torch
from llm_response_models import KeywordExtractionResponse
from prompt_templates import get_keyword_prompt
from project_config import KEYWORD_EXTRACTION_PRESETS as PRESETS
from project_config import KeywordExtractionConfig

logger = logging.getLogger(__name__)


class KeywordExtractor:
    def __init__(self, model, tokenizer):
        """
        Initialize the keyword extractor with a local model and tokenizer.
        Assumes both are already loaded and moved to the correct device.
        """
        self.model = model
        self.tokenizer = tokenizer

    def extract(
        self,
        text: str,
        max_keywords: int = 40,
        mode: str = "balanced",
        prompt_type: str = "default",
        max_new_tokens: Optional[int] = None,
    ) -> KeywordExtractionResponse:
        """
        Main entrypoint: runs LLM keyword extraction with parsing and fallback handling.

        Args:
            text (str): Input text for extraction.
            max_keywords (int): Max number of keywords to extract (not enforced yet).
            mode (str): Preset config mode.
            prompt_type (str): Prompt template to use.
            max_new_tokens (int): Override for token generation length.

        Returns:
            KeywordExtractionResponse: Structured response with keywords or error info.
        """
        start_time = time.time()

        if not text.strip():
            return self._error_result("Empty input text", 0.0)

        try:
            config = PRESETS.get(mode, PRESETS["balanced"])

            if max_new_tokens is not None:
                config = KeywordExtractionConfig(
                    temperature=config.temperature,
                    top_k=config.top_k,
                    top_p=config.top_p,
                    max_new_tokens=max_new_tokens,
                )

            raw_output = self._generate_with_config(text, config, prompt_type)
            logger.debug(f"[Raw model output]\n{raw_output}")

            keywords = self._parse_keywords(raw_output)
            logger.debug(f"[Extracted keywords] {keywords}")

            return KeywordExtractionResponse(
                keywords=keywords,
                status="success",
                processing_time_sec=time.time() - start_time,
            )

        except Exception as e:
            logger.error(f"Extraction failed: {str(e)}", exc_info=True)
            return self._error_result(str(e), time.time() - start_time)

    def _generate_with_config(self, text: str, config, prompt_type: str) -> str:
        """
        Generates keyword extraction response from the model.

        Args:
            text (str): Source text.
            config: KeywordExtractionConfig object.
            prompt_type (str): Name of the prompt template to use.

        Returns:
            str: Raw decoded LLM output.
        """
        prompt = get_keyword_prompt(text, prompt_type)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        logger.info(f"[Prompt for {prompt_type} mode]:\n{prompt}")

        torch.manual_seed(42)  # Ensures repeatability

        output = self.model.generate(
            **inputs,
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature,
            top_k=config.top_k,
            top_p=config.top_p,
            do_sample=True,
        )[0]

        return self.tokenizer.decode(output, skip_special_tokens=True)

    def _parse_keywords(self, raw_output: str) -> List[str]:
        """
        Parses keywords from raw model output using multiple strategies:
        - Preferred: content between [KEYWORDS]...[/KEYWORDS] after </think>
        - Fallback: [kw1, kw2, ...] anywhere
        - Heuristic: final line comma-separated list

        Args:
            raw_output (str): Full LLM output.

        Returns:
            List[str]: Cleaned keyword list or fallback result.
        """
        logger.info(f"[Raw text for parsing]\n{raw_output}")

        # Step 1: Try to extract after </think> if present
        try:
            *_, model_output = raw_output.rsplit("</think>", 1)
        except ValueError:
            model_output = raw_output  # Fallback if no </think>

        # * Step 2: Manually search for the last occurrence of "[KEYWORDS]" and its closing tag.
        start_index = model_output.rfind("[KEYWORDS]")
        if start_index != -1:
            end_index = model_output.find("[/KEYWORDS]", start_index)
            if end_index != -1:
                keyword_str = model_output[
                    start_index + len("[KEYWORDS]") : end_index
                ].strip()

                # Try parsing as an actual Python list first.
                try:
                    parsed_list = ast.literal_eval(keyword_str)
                    if isinstance(parsed_list, list):
                        return [
                            kw.strip()
                            for kw in parsed_list
                            if isinstance(kw, str) and not self._is_placeholder(kw)
                        ]
                except (SyntaxError, ValueError):
                    pass

                # Fallback: manual comma-split
                keywords = [
                    kw.strip()
                    for kw in keyword_str.split(",")
                    if kw.strip() and not self._is_placeholder(kw)
                ]
                if keywords:
                    return keywords

        # * Step 3: Parse from [KEYWORDS]...[/KEYWORDS] using the last occurrence
        # re.findall(...) returns all non-overlapping matches of the pattern in the string as a list.
        matches = re.findall(r"\[KEYWORDS\](.*?)\[/KEYWORDS\]", model_output, re.DOTALL)

        if matches:
            keyword_str = matches[-1].strip()

            # Try parsing as actual Python list first ["...", "...", ...]
            try:
                parsed_list = ast.literal_eval(keyword_str)
                if isinstance(parsed_list, list):
                    return [
                        kw.strip()
                        for kw in parsed_list
                        if isinstance(kw, str) and not self._is_placeholder(kw)
                    ]
            except (SyntaxError, ValueError):
                pass

            # Fallback: manual comma-split
            keywords = [
                kw.strip()
                for kw in keyword_str.split(",")
                if kw.strip() and not self._is_placeholder(kw)
            ]
            if keywords:
                return keywords

        # Step 4: Parse generic [term1, term2] format
        elif match := re.search(r"\[(.*?)\]", model_output, re.DOTALL):
            keyword_str = match.group(1).strip()
            try:
                parsed_list = ast.literal_eval(f"[{keyword_str}]")
                if isinstance(parsed_list, list):
                    return [
                        kw.strip()
                        for kw in parsed_list
                        if isinstance(kw, str) and not self._is_placeholder(kw)
                    ]
            except (SyntaxError, ValueError):
                pass

            keywords = [
                kw.strip()
                for kw in keyword_str.split(",")
                if kw.strip() and not self._is_placeholder(kw)
            ]
            if keywords:
                return keywords

        # Step 5: Fallback - last line comma-split heuristic
        last_line = model_output.strip().split("\n")[-1].strip()
        if "," in last_line:
            keywords = [
                kw.strip()
                for kw in last_line.split(",")
                if kw.strip() and not self._is_placeholder(kw)
            ]
            if len(keywords) > 1:
                return keywords

        return self._fallback_parse()

    def _is_placeholder(self, text: str) -> bool:
        """
        Returns True if the keyword is clearly a placeholder or invalid.

        Args:
            text (str): Keyword candidate.

        Returns:
            bool: True if placeholder-like.
        """
        placeholders = {
            "keyword1",
            "keyword2",
            "example",
            "kw1",
            "kw2",
            "noun1",
            "noun2",
        }
        return (
            text.lower() in placeholders or len(text) < 2 or text.startswith(("[", "("))
        )

    def _fallback_parse(self) -> List[str]:
        """
        Final fallback when no structured keywords are found.

        Returns:
            List[str]: Sentinel result indicating failure.
        """
        logger.warning("Keyword extraction failed — no valid format found.")
        return ["[NO_KEYWORDS_FOUND]"]

    def _error_result(self, message: str, duration: float) -> KeywordExtractionResponse:
        """
        Standardized error response wrapper.

        Args:
            message (str): Error message.
            duration (float): Processing time in seconds.

        Returns:
            KeywordExtractionResponse
        """
        return KeywordExtractionResponse(
            keywords=[],
            status="error",
            message=message,
            processing_time_sec=duration,
        )
