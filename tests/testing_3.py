import re

raw_output = """
    Extract technical terms from: "Fermentation is a metabolic process where microorganisms like yeast and lactic acid bacteria convert sugars into alcohol, acids, or gases. Key techniques include sourdough bread-making (using wild yeast), kimchi production (via lacto-fermentation), and brewing (where Saccharomyces cerevisiae ferments malted grains). Factors like temperature, pH, and salinity critically impact microbial activity. Common challenges include contamination, inconsistent fermentation rates, and off-flavors from byproducts."

    <think>
    1. Identify ONLY the core technical terms (ignore descriptions, examples, verbs).  
    2. Keywords must be:  
    - Nouns or noun phrases (e.g., "neural networks," not "excel at pattern recognition").  
    - Directly mentioned in the text (no paraphrasing).  
    3. Output MUST follow this format:  
    [KEYWORDS]keyword1, keyword2, keyword3[/KEYWORDS]  
    - No extra text, explanations, or original sentences.  
    </think>
     [KEYWORDS]Key techniques include sourdough bread-making (using wild yeast), kimchi production (via lacto-fermentation), and brewing (where Saccharomyces cerevisiae ferments malted grains).[/KEYWORDS]
"""

matches = list(
    re.finditer(
        r"\[KEYWORDS\](.*?)\[/KEYWORDS\]", raw_output, re.DOTALL | re.IGNORECASE
    )
)

for match in matches:
    print("Extracted:", match.group(1))
