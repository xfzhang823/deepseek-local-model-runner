# Configure logging with the application name "DeepSeek"
import logging
from logging_config import configure_logging

configure_logging(app_name="DeepSeek")

# Standard logger
logger = logging.getLogger(__name__)
logger.info("Application started")

# Resource logger
resource_logger = logging.getLogger("resource")
resource_logger.info("Resource usage: CPU 45%, RAM 1.5GB, GPU VRAM 256MB")
