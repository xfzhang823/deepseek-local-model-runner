import sys
import os
import logging
import logging_config
from loaders.awq_loader import AWQ_ModelLoader
from safetensors.torch import load_file
from pipelines.scrooge_quant_pipeline import run_scrooge_quant_pipeline
from project_config import DEEPSEEK_R1_DISTILL_QUANT_MODEL_OUTPUT_DIR
import logging_config

logger = logging.getLogger(__name__)
resource_logger = logging.getLogger("resource")


# state_dict = load_file("/home/xzhang/models/deepseek-awq/model.safetensors")
# for key, tensor in state_dict.items():
#     logger.info(
#         f"{key}: shape={tensor.shape}, dtype={tensor.dtype}, device={tensor.device}"
#     )


# def main():
#     tokenizer, model = AWQ_ModelLoader.load_model()
#     logger.info("✅ AWQ model and tokenizer loaded successfully!")
#     logger.info(f"Model device: {next(model.parameters()).device}")
#     logger.info(f"Tokenizer vocab size: {tokenizer.vocab_size}")


def main():
    # Define your input model path and save path
    model_path = str(DEEPSEEK_R1_DISTILL_QUANT_MODEL_OUTPUT_DIR)
    save_path = str(DEEPSEEK_R1_DISTILL_QUANT_MODEL_OUTPUT_DIR)
    save_layers_dir = str(
        DEEPSEEK_R1_DISTILL_QUANT_MODEL_OUTPUT_DIR / "quantized_layers_duo_scaling"
    )  # if using duo_sclaing

    # Run quantization pipeline
    run_scrooge_quant_pipeline(
        # save_dir_path=save_path,
        max_calib_samples=48,
        max_calib_seq_len=768,  # Set it higher for the reasoning model
        apply_clip=True,
    )

    logger.info("✅ Quantization pipeline completed successfully.")


if __name__ == "__main__":
    main()
