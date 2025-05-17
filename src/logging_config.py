"""
logging_config.py

This module provides a centralized and customizable logging setup for applications.
It supports logging to both a single file and the console, with detailed formatting
and log rotation. Additionally, a separate session-based `resource.log` file is
maintained for resource monitoring (e.g., CPU, RAM, GPU usage).

Usage:
    >>> from logging_config import configure_logging
    >>> configure_logging(app_name="my_app")

    >>> import logging
    >>> logger = logging.getLogger(__name__)
    >>> logger.info("This is an info message")

Version: 3.1
Author: Xiao-Fei Zhang
"""

import logging
import logging.handlers
import os
import getpass
from datetime import datetime
from pathlib import Path


def find_project_root(starting_path=None, marker=".git"):
    "Find and return directory path of the project root dir."
    if starting_path is None:
        starting_path = Path(__file__).resolve().parent
    starting_path = Path(starting_path)
    for parent in [starting_path] + list(starting_path.parents):
        if (parent / marker).exists():
            return parent
    return None


class EnhancedLoggerSetup:
    """
    Class to centralize and standardize logging setup for the entire application.
    It handles the initialization of loggers, file paths, and handlers for both regular logs
    and resource logs.
    """

    def __init__(self, app_name=None, debug=True):
        self.app_name = app_name or "app"
        self.root_dir = find_project_root()
        if self.root_dir is None:
            raise RuntimeError("Project root not found (missing .git marker)")
        self.logs_dir = os.path.join(self.root_dir, "logs")
        os.makedirs(self.logs_dir, exist_ok=True)

        self.username = getpass.getuser()
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.debug = debug

        # Initialize formatters
        self.detailed_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(filename)s:%(lineno)d - %(levelname)s - %(message)s"
        )
        self.console_formatter = logging.Formatter("%(levelname)s - %(message)s")

    def get_log_file_path(self):
        """Path for the main log file (everything except resource monitoring)."""
        filename = f"{self.username}_{self.session_id}.log"
        return os.path.join(self.logs_dir, filename)

    def get_resource_log_path(self):
        """Generate a session-based resource log filename."""
        filename = f"{self.username}_{self.session_id}_resource.log"
        return os.path.join(self.logs_dir, filename)

    def setup_file_handler(self):
        """Setup the main file handler."""
        file_handler = logging.handlers.RotatingFileHandler(
            self.get_log_file_path(), maxBytes=10 * 1024 * 1024, backupCount=5
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(self.detailed_formatter)
        return file_handler

    def setup_console_handler(self):
        """Setup the console handler."""
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(self.console_formatter)
        return console_handler

    def setup_resource_handler(self):
        """Setup a separate resource log handler with session-based naming."""
        resource_handler = logging.handlers.RotatingFileHandler(
            self.get_resource_log_path(), maxBytes=5 * 1024 * 1024, backupCount=3
        )
        resource_handler.setLevel(logging.INFO)
        resource_handler.setFormatter(self.detailed_formatter)
        return resource_handler

    def configure_logging(self):
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)

        # Clear existing handlers to prevent duplicates
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)

        # Main log handlers
        logger.addHandler(self.setup_file_handler())
        logger.addHandler(self.setup_console_handler())

        # Resource log handler
        resource_logger = logging.getLogger("resource")
        resource_logger.setLevel(logging.INFO)
        resource_logger.addHandler(self.setup_resource_handler())
        resource_logger.propagate = False

        logger.info(
            "✅ Logging configured successfully (console + file + resource.log)"
        )


def configure_logging(app_name=None, debug=True):
    """Configure logging with optional app name and debug mode."""
    logger_setup = EnhancedLoggerSetup(app_name, debug=debug)
    logger_setup.configure_logging()


# Automatically configure when imported
if __name__ != "__main__":
    configure_logging()
