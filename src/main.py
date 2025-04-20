import logging
from task_router import run_llm_task
from api import translate_batch
from llm_response_models import (
    SummarizationResponse,
    TranslationResponse,
    KeywordExtractionResponse,
    TopicGenerationResponse,
    TextAlignmentResponse,
)
import logging_config

logger = logging.getLogger(__name__)


def test_summarization():
    result = run_llm_task(
        "summarization",
        text="""
        Large language models (LLMs) like GPT, Claude, and DeepSeek are advanced neural networks trained on massive corpora of text data. 
        They are capable of generating coherent and contextually appropriate responses across a wide range of topics, including code, essays, and even poetry. 
        These models use transformer-based architectures and rely heavily on techniques like self-attention to process sequences of text and predict the next word or phrase.

        One of the most powerful features of LLMs is their ability to generalize knowledge from their training data and apply it to new, unseen prompts. 
        This makes them useful for tasks such as summarization, translation, topic generation, and alignment of text. 
        Despite their strengths, LLMs have limitations — they can hallucinate facts, produce biased outputs, or misinterpret ambiguous instructions.

        As LLMs continue to evolve, researchers are focusing on improving their accuracy, reducing model size without sacrificing performance, and enabling more efficient local deployment to minimize cost and improve privacy.
        """,
    )
    print("\n[Test: Summarization]")
    print(result)


def test_translation():
    result = run_llm_task(
        "translation", text="How are you today?", target_lang="Spanish"
    )
    print("\n[Test: Translation]")
    print(result)


text_1 = "Neural networks, reinforcement learning, and supervised learning are major ML subfields, \
each with distinct approaches: neural networks excel at pattern recognition through layered architectures, \
reinforcement learning optimizes decision-making via reward systems, and supervised learning relies on \
labeled datasets to train predictive models."

text_2 = "Deep learning architectures such as convolutional neural networks (CNNs) and recurrent neural \
networks (RNNs) have transformed computer vision and natural language processing (NLP). CNNs leverage hierarchical \
feature extraction through convolutional layers, while RNNs use sequential memory cells like LSTMs to handle \
time-series data. Meanwhile, transformer models like BERT and GPT-4 rely on self-attention mechanisms, \
enabling parallel processing and long-range dependency capture. Traditional methods like SVMs and decision \
trees remain useful for smaller datasets, but they lack the scalability of deep learning approaches. \
Key challenges include overfitting, computational costs, and the need for large labeled datasets."

text_3 = "Fermentation is a metabolic process where microorganisms like yeast and lactic acid bacteria convert \
sugars into alcohol, acids, or gases. Key techniques include sourdough bread-making (using wild yeast), \
kimchi production (via lacto-fermentation), and brewing (where Saccharomyces cerevisiae ferments malted grains). \
Factors like temperature, pH, and salinity critically impact microbial activity. Common challenges include \
contamination, inconsistent fermentation rates, and off-flavors from byproducts."


def test_keyword_extraction():
    result = run_llm_task("keyword_extraction", text=text_1)
    print("\n[Test: Keyword Extraction]")
    print(result)


def test_topic_generation():
    result = run_llm_task(
        "topic_generation",
        text="Neural networks, reinforcement learning, and supervised learning are major ML subfields, \
each with distinct approaches: neural networks excel at pattern recognition through layered architectures, \
reinforcement learning optimizes decision-making via reward systems, and supervised learning relies on \
labeled datasets to train predictive models.",
    )
    print("\n[Test: Topic Generation]")
    print(result)


def test_text_alignment():
    source = "Our product is fast, reliable, and secure."
    target = "This service is secure and high-performing."
    result = run_llm_task("text_alignment", source_text=source, target_text=target)
    print("\n[Test: Text Alignment]")
    print(result)


def test_translation_batch():
    texts = [
        "The weather is nice today.",
        "I love programming in Python.",
        "Artificial intelligence is transforming industries.",
        "What time is the meeting tomorrow?",
        "Thank you for your help!",
    ]

    result = translate_batch(texts, target_lang="German")
    print("\n[Test: Translation Batch]")
    for i, res in enumerate(result, 1):
        print(f"[{i}] {res.translated_text} (status: {res.status})")


def main():
    print("🔍 Running test tasks...")

    # test_summarization()  # this works
    # test_translation() # this works
    test_keyword_extraction()  # this works
    # test_topic_generation() # * this does not work yet (but I am not sure exactly it provides much value)
    # test_text_alignment()
    # test_translation_batch()


if __name__ == "__main__":
    main()
