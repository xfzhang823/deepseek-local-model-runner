"""prompts.py"""

from typing import Dict

# Prompt templates for various LLM tasks, with strict formatting rules.

SUMMARIZATION_PROMPT = """
Please summarize the following text in 1-3 sentences.

STRICT INSTRUCTIONS:
1. Your response must contain ONLY the summary.
2. Format EXACTLY like this: <summary>summary text here</summary>
3. Do NOT include any other text, analysis, or tags.

Text to summarize:
{text}
"""

TEXT_ALIGNMENT_PROMPT = """
Compare the following texts and align their main ideas.

STRICT INSTRUCTIONS:
1. Your response must contain ONLY the alignment.
2. Format EXACTLY like this: <alignment>Aligned idea pairs or mappings</alignment>
3. Do NOT include any other text, analysis, or tags.

Source:
{source_text}

Target:
{target_text}
"""

TOPIC_GENERATION_PROMPT = """
Extract key topics from the following text as comma-separated values.

STRICT INSTRUCTIONS:
1. Your response must contain ONLY the topics.
2. Only include concepts DIRECTLY from this text: {text}
3. Do NOT include examples or instructions.
4. Format EXACTLY like this: <topics>topic1, topic2</topics>

Text:
{text}
"""

TRANSLATION_PROMPT = """
Translate the following text into {target_lang} using natural, colloquial language.

STRICT INSTRUCTIONS:
1. Your response must contain ONLY the translation.
2. Format EXACTLY like this: <translated>TRANSLATION_HERE</translated>
3. Do NOT include the original text, explanations, or notes.
4. Do NOT include any text outside the <translated> tags.

Text to translate:
{text_to_translate}
"""
