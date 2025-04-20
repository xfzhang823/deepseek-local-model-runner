"""prompt_templates.py"""

from typing import Dict

#    - NO extra text, explanation, or commentary between [KEYWORDS] and [/KEYWORDS].

KEYWORD_PROMPTS: Dict[str, str] = {
    "default": """
    Extract the most relevant keywords from: "{text}"

    <think>
    1. Reason through step by step without repeating yourself. \
Each step should be concise and progress logically toward the final answer. 
    2. Analyze the text and identify ONLY the core keywords:  
       - Nouns/noun phrases, directly mentioned (no paraphrasing).
    3. Keyword output format must be a Python-compatible list of strings:
       - Enclosed in [KEYWORDS][...][/KEYWORDS] for programmatic parsing.  
       - Example: [KEYWORDS][keyword1, keyword2][/KEYWORDS].
    4. Why this format?  
       - Your output will be validated by a Pydantic model (expects `list[str]`).  
       - Stray text or deviations will cause parsing failures. 
    """,
    "technical": """
    Extract technical terms from: "{text}"

    <think>
    1. Please reason through the problem step by step without repeating yourself. \
Each step should be concise and progress logically toward the final answer. 
    2. Identify ONLY the core technical terms (ignore descriptions, examples, verbs).  
    3. Keywords must be:  
    - Nouns or noun phrases (e.g., "neural networks," not "excel at pattern recognition").  
    - Directly mentioned in the text (no paraphrasing).  
    4. *MUST place the extracted keywords in follow this format*:  
    [KEYWORDS]keyword1, keyword2, keyword3[/KEYWORDS]  
    - No extra text, explanations, or original sentences between [KEYWORDS] and [/KEYWORDS]
    - Reason: I am processing your output with a Pydantic model with an attribute keywords: List[str]; \
therefore, I need easily parse out the keywords with the [KEYWORDS] and [/KEYWORDS] tags.
    """,
}

# without asking for Python list, just Dict of List
KEYWORD_PROMPTS_0A: Dict[str, str] = {
    "default": """
    Extract the most relevant keywords from: "{text}"

    <think>
    1. Reason through step by step without repeating yourself. \
Each step should be concise and progress logically toward the final answer. 
    2. Analyze the text and identify ONLY the core keywords:  
       - Nouns/noun phrases, directly mentioned (no paraphrasing).
    3. Keyword output format must be a list of strings:
       - Enclosed in [KEYWORDS][...][/KEYWORDS] for programmatic parsing.  
       - Example: [KEYWORDS][keyword1, keyword2][/KEYWORDS].
    4. Why this format?  
       - Your output will be validated by a Pydantic model (expects `list[str]`).  
       - Stray text or deviations will cause parsing failures. 
    """,
    "technical": """
    Extract technical terms from: "{text}"

    <think>
    1. Please reason through the problem step by step without repeating yourself. \
Each step should be concise and progress logically toward the final answer. 
    2. Identify ONLY the core technical terms (ignore descriptions, examples, verbs).  
    3. Keywords must be:  
    - Nouns or noun phrases (e.g., "neural networks," not "excel at pattern recognition").  
    - Directly mentioned in the text (no paraphrasing).  
    4. *MUST place the extracted keywords in follow this format*:  
    [KEYWORDS]keyword1, keyword2, keyword3[/KEYWORDS]  
    - No extra text, explanations, or original sentences between [KEYWORDS] and [/KEYWORDS]
    - Reason: I am processing your output with a Pydantic model with an attribute keywords: List[str]; \
therefore, I need easily parse out the keywords with the [KEYWORDS] and [/KEYWORDS] tags.
    """,
}


KEYWORD_PROMPTS_0B: Dict[str, str] = {
    "default": """
    Extract the most relevant keywords from: "{text}"

    <think>
    1. Reason through step by step without repeating yourself. \
Each step should be concise and progress logically toward the final answer. 
    2. Analyze the text and identify ONLY the core keywords:  
       - Nouns/noun phrases, directly mentioned (no paraphrasing).
    3. Keyword output format must be a Python-compatible list of strings:
       - Enclosed in [KEYWORDS][...][/KEYWORDS] for programmatic parsing.  
       - Example: [KEYWORDS][keyword1, keyword2][/KEYWORDS].
       - NO extra text, explanation, or commentary between [KEYWORDS] and [/KEYWORDS].
    """,
    "technical": """
    Extract technical terms from: "{text}"

    <think>
    1. Please reason through the problem step by step without repeating yourself. \
Each step should be concise and progress logically toward the final answer. 
    2. Identify ONLY the core technical terms (ignore descriptions, examples, verbs).  
    3. Keywords must be:  
    - Nouns or noun phrases (e.g., "neural networks," not "excel at pattern recognition").  
    - Directly mentioned in the text (no paraphrasing).  
    4. *MUST place the extracted keywords in follow this format*:  
    [KEYWORDS]keyword1, keyword2, keyword3[/KEYWORDS]  
    - No extra text, explanations, or original sentences between [KEYWORDS] and [/KEYWORDS]
    - Reason: I am processing your output with a Pydantic model with an attribute keywords: List[str]; \
therefore, I need easily parse out the keywords with the [KEYWORDS] and [/KEYWORDS] tags.
    """,
}

KEYWORD_PROMPTS_0C: Dict[str, str] = {
    "default": """
    Extract the most relevant keywords from: "{text}"

    <think>
    1. Reason through the problem step by step without repeating yourself. \
Each step should be concise and progress logically toward the final answer. 
    2. Identify ONLY the core terms...
    3. Keywords must be:
    - Nouns or noun phrases...
    - Directly mentioned in the text (no paraphrasing). 
    4. Output the keywords ONLY inside the tags below, in Python list-of-strings format:
        [KEYWORDS]["keyword1", "keyword2", "keyword3"][/KEYWORDS]
        NO extra text, explanation, or commentary between [KEYWORDS] and [/KEYWORDS].
    """,
    "technical": """
    Extract technical terms from: "{text}"

    <think>
    1. Please reason through the problem step by step without repeating yourself. \
Each step should be concise and progress logically toward the final answer. 
    2. Identify ONLY the core technical terms (ignore descriptions, examples, verbs).  
    3. Keywords must be:  
    - Nouns or noun phrases (e.g., "neural networks," not "excel at pattern recognition").  
    - Directly mentioned in the text (no paraphrasing).  
    4. *MUST place the extracted keywords in follow this format*:  
    [KEYWORDS]keyword1, keyword2, keyword3[/KEYWORDS]  
    - No extra text, explanations, or original sentences between [KEYWORDS] and [/KEYWORDS]
    - Reason: I am processing your output with a Pydantic model with an attribute keywords: List[str]; \
therefore, I need easily parse out the keywords with the [KEYWORDS] and [/KEYWORDS] tags.
    """,
}

# todo: this prompt does not work - infinite reasoning loop; delete later after fully debugged
KEYWORD_PROMPTS_1: Dict[str, str] = {
    "default": """
    Extract the most relevant keywords from: "{text}"

    <think>
    1. Identify ONLY the core technical terms...
    2. Keywords must be:
    - Nouns or noun phrases...
    3. Output MUST follow this format:
    [KEYWORDS]...[/KEYWORDS]
    </think>
    """,
    "technical": """
    Extract technical terms from: "{text}"

    <think>
    1. Reason step-by-step, keeping each step concise and unique.
    2. Identify ONLY nouns or noun phrases that are domain-specific technical terms (e.g., "database", "algorithm").
    3. Use only terms explicitly in the text—no paraphrasing.
    4. Format output as: [KEYWORDS]keyword1, keyword2, keyword3[/KEYWORDS]
    </think>

    Output rules:
    - Place keywords in [KEYWORDS]...[/KEYWORDS], separated by commas and optional spaces.
    - No extra words, sentences, or newlines between [KEYWORDS] and [/KEYWORDS]—just the comma-separated list.
    - I’ll extract this directly for a Pydantic model, so keep it simple. Text outside the tags is fine.
    """,
}

KEYWORD_PROMPTS_2: Dict[str, str] = {
    "default": """
    Extract the most relevant keywords from: "{text}"

    <think>
    1. Identify ONLY the core technical terms...
    2. Keywords must be:
    - Nouns or noun phrases...
    3. Output MUST follow this format:
    [KEYWORDS]...[/KEYWORDS]
    </think>
    """,
    "technical": """
    Extract technical terms from: "{text}"

    Instructions:
    - Identify nouns or noun phrases that are domain-specific technical terms (e.g., "fermentation", "yeast", not "convert").
    - Use only terms directly in the text—no paraphrasing.
    - Output as: [KEYWORDS]keyword1, keyword2, keyword3[/KEYWORDS]
    - No text between [KEYWORDS] and [/KEYWORDS]—just the comma-separated list.
    - Extra text outside the tags is okay; I’ll parse the keywords for a Pydantic model.
        """,
}

KEYWORD_PROMPTS_3: Dict[str, str] = {
    "default": """
    Extract the most relevant keywords from: "{text}"

    <think>
    1. Identify ONLY the core terms...
    2. Keywords must be:
    - Nouns or noun phrases...
    3. Output MUST follow this format:
    [KEYWORDS]...[/KEYWORDS]
    </think>
    """,
    "technical": """
    Extract technical terms from the text below. You *MUST* format the output as:  
    [KEYWORDS]comma_separated_list_of_terms[/KEYWORDS]  

    Text: "{text}" 

    [KEYWORDS]
    """,
}


def get_keyword_prompt(text: str, prompt_type: str = "default") -> str:
    return KEYWORD_PROMPTS[prompt_type].format(text=text[:2000])  # Truncate long text
