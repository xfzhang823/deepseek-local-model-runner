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

texts_dict = {
    "text_1": "Neural networks, reinforcement learning, and supervised learning are major ML subfields, each with distinct approaches: neural networks excel at pattern recognition through layered architectures, reinforcement learning optimizes decision-making via reward systems, and supervised learning relies on labeled datasets to train predictive models.",
    "text_2": "Deep learning architectures such as convolutional neural networks (CNNs) and recurrent neural networks (RNNs) have transformed computer vision and natural language processing (NLP). CNNs leverage hierarchical feature extraction through convolutional layers, while RNNs use sequential memory cells like LSTMs to handle time-series data. Meanwhile, transformer models like BERT and GPT-4 rely on self-attention mechanisms, enabling parallel processing and long-range dependency capture. Traditional methods like SVMs and decision trees remain useful for smaller datasets, but they lack the scalability of deep learning approaches. Key challenges include overfitting, computational costs, and the need for large labeled datasets.",
    "text_3": "Fermentation is a metabolic process where microorganisms like yeast and lactic acid bacteria convert sugars into alcohol, acids, or gases. Key techniques include sourdough bread-making (using wild yeast), kimchi production (via lacto-fermentation), and brewing (where Saccharomyces cerevisiae ferments malted grains). Factors like temperature, pH, and salinity critically impact microbial activity. Common challenges include contamination, inconsistent fermentation rates, and off-flavors from byproducts.",
    "text_4": "Neural networks and unsupervised learning techniques play a pivotal role in modern AI applications. Algorithms such as autoencoders and clustering methods enable systems to discover hidden structures in data. While neural networks excel in capturing complex patterns through multi-layer architectures, unsupervised methods are particularly valuable when labeled datasets are scarce. Key challenges include tuning hyperparameters, avoiding local minima, and ensuring model generalization.",
    "text_5": "Recent advancements in deep learning have revolutionized image and speech recognition systems. Convolutional layers in deep architectures are adept at extracting spatial hierarchies, while recurrent networks capture temporal dependencies in sequential data. Moreover, attention mechanisms further enhance the performance of these models by focusing on relevant features. Despite their success, these models require large datasets and significant computational resources.",
    "text_6": "Biotechnological processes such as fermentation continue to impact various industries, from food production to pharmaceuticals. Microorganisms like bacteria and yeast are harnessed to transform organic substrates into valuable products including biofuels, enzymes, and organic acids. Factors such as temperature, pH, and nutrient availability must be carefully controlled to optimize yields. Innovations in bioreactor design and process automation are helping to overcome traditional limitations.",
    "text_7": "The global economy is undergoing significant shifts as emerging markets gain momentum. Analysts note that changes in trade policies, fluctuating commodity prices, and evolving investor sentiments are reshaping financial landscapes. Key factors include inflation trends, interest rate adjustments, and political uncertainties that affect market stability and growth prospects.",
    "text_8": "In the world of sports, teams are adapting strategies to outmaneuver opponents in fast-paced competitions. The latest season has seen a surge in innovative playstyles, rigorous training regimes, and an increased emphasis on data analytics to enhance performance. Challenges remain in maintaining player health, team cohesion, and strategic versatility.",
    "text_9": "Recent developments in healthcare have led to breakthroughs in personalized medicine and gene therapy. Medical researchers are focusing on early diagnostics and targeted treatment strategies to address complex diseases. With the rise of telemedicine and digital health tools, patient care is becoming more accessible, though challenges in regulatory approval and data privacy persist.",
    "text_10": "Literary trends are evolving with the advent of digital publishing and global connectivity. Contemporary authors explore themes of identity, technology, and socio-political change through diverse narrative styles. Literary festivals and book fairs are fostering cross-cultural exchanges, while critics debate the impact of digital media on traditional storytelling techniques.",
    "text_11": "Environmental science is at the forefront of addressing climate change and sustainability challenges. Researchers are examining the effects of greenhouse gas emissions, deforestation, and urbanization on ecosystems. Renewable energy projects, conservation efforts, and innovative waste management practices are central to reducing the environmental footprint and promoting long-term ecological balance.",
}


def test_keyword_extraction():
    results_summary = {}
    cap = 15  # Maximum number of keywords to show
    for key, text in texts_dict.items():
        result = run_llm_task("keyword_extraction", text=text)
        # Extract status and keywords if available; else default.
        status = getattr(result, "status", "unknown")
        keywords = getattr(result, "keywords", [])
        # Cap the keyword list output
        keywords_capped = keywords[:cap]
        results_summary[key] = {"status": status, "keywords": keywords_capped}
        logger.info(f"\n[Test: Keyword Extraction for {key}]")
        logger.info(result)

    # Log the summary sheet with both status and capped keywords.
    logger.info("\n=== Keyword Extraction Summary ===")
    for key, summary in results_summary.items():
        logger.info(
            f"{key}: Status: {summary['status']} | Keywords: {summary['keywords']}"
        )


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
