import logging
from pipelines.awq_quant_pipeline import run_awq_quant_pipeline
import logging_config

logger = logging.getLogger(__name__)


def main():
    run_awq_quant_pipeline()


if __name__ == "__main__":
    main()
