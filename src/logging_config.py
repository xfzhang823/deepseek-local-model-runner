"""
Logging Configuration Module

This module provides a centralized and customizable logging setup for applications.
It supports logging to both a single file and the console, with detailed formatting
and log rotation. Logs are organized by session, with each session generating a
unique log file.

Features:
- Session-based log files (e.g., `username_YYYYMMDD_HHMMSS.log`).
- Rotating file handler with configurable size and backup count.
- Detailed log formatting including timestamp, module, filename, line number,
and log level.
- Console logging with simplified formatting for real-time monitoring.

Usage:
    >>> from logging_config import configure_logging
    >>> configure_logging(app_name="my_app")

    >>> import logging
    >>> logger = logging.getLogger(__name__)
    >>> logger.info("This is an info message")

Version: 2.0
Author: Xiao-Fei Zhang
"""

import logging
import logging.handlers
import os
import getpass
from datetime import datetime
from pathlib import Path


def find_project_root(starting_path=None, marker=".git"):
    if starting_path is None:
        starting_path = Path(__file__).resolve().parent
    starting_path = Path(starting_path)
    for parent in [starting_path] + list(starting_path.parents):
        if (parent / marker).exists():
            return parent
    return None


class EnhancedLoggerSetup:
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
        filename = f"{self.username}_{self.session_id}.log"
        return os.path.join(self.logs_dir, filename)

    def setup_file_handler(self):
        file_handler = logging.handlers.RotatingFileHandler(
            self.get_log_file_path(), maxBytes=10 * 1024 * 1024, backupCount=5
        )
        file_handler.setLevel(logging.DEBUG)  # 🔥 Everything
        file_handler.setFormatter(self.detailed_formatter)
        return file_handler

    def setup_console_handler(self):
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)  # 🔥 Everything
        console_handler.setFormatter(self.console_formatter)
        return console_handler

    def configure_logging(self):
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)  # 🔥 Root logger DEBUG level

        # Clear any existing handlers
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)

        file_handler = self.setup_file_handler()
        console_handler = self.setup_console_handler()

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        logger.propagate = False

        logger.info("✅ Logging configured successfully (console + file both DEBUG)")


def configure_logging(app_name=None, debug=True):
    logger_setup = EnhancedLoggerSetup(app_name, debug=debug)
    logger_setup.configure_logging()


# Automatically configure when imported
if __name__ != "__main__":
    configure_logging()
