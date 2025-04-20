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
    """
    Recursively find the root directory of the project by looking for a specific marker.

    Args:
        starting_path (str or Path): The starting path to begin the search. Defaults to
        the current script's directory.
        marker (str): The marker to look for (e.g., '.git', 'setup.py', 'README.md').

    Returns:
        Path: The Path object pointing to the root directory of the project,
        or None if not found.
    """
    if starting_path is None:
        starting_path = Path(__file__).resolve().parent

    starting_path = Path(starting_path)

    print(f"\n🔍 Searching for project root starting from: {starting_path}")

    # ✅ First, check if the marker exists in the current directory
    marker_path = starting_path / marker
    print(f"🟡 Checking: {marker_path} → Exists? {marker_path.exists()}")

    if marker_path.exists():
        print(f"✅ Found marker in starting directory: {starting_path}")
        return starting_path

    # ✅ Then, check parent directories
    for parent in starting_path.parents:
        marker_path = parent / marker
        print(f"🟡 Checking: {marker_path} → Exists? {marker_path.exists()}")

        if marker_path.exists():
            print(f"✅ Found project root at: {parent}")
            return parent

    print("❌ Project root not found!")
    return None


class EnhancedLoggerSetup:
    def __init__(self, app_name=None):
        """
        Initialize logging setup with application name and logging directories.

        Args:
            app_name (str, optional): Name of your application.
        """
        self.app_name = app_name or "app"
        self.root_dir = find_project_root()
        if self.root_dir is None:
            raise RuntimeError(
                "Project root not found. Ensure the marker (e.g., .git) is present."
            )
        self.logs_dir = os.path.join(self.root_dir, "logs")
        self.username = self.get_username()
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")  # Unique session ID

        # Create logs directory if it doesn't exist
        os.makedirs(self.logs_dir, exist_ok=True)

        # Initialize formatters
        self.detailed_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(filename)s:%(lineno)d - %(levelname)s - %(message)s"
        )
        self.console_formatter = logging.Formatter(
            "%(name)s - %(levelname)s - %(message)s"
        )

    def get_username(self):
        """
        Retrieves the current username.

        Returns:
            str: Username of the current user.
        """
        username = getpass.getuser()
        print(f"[DEBUG] Retrieved username: {username}")
        return username

    def get_log_file_path(self):
        """
        Constructs the log file path based on the username and session ID.

        Returns:
            str: Log file path.
        """
        filename = f"{self.username}_{self.session_id}.log"
        log_path = os.path.join(self.logs_dir, filename)
        print(f"[DEBUG] Log file path: {log_path}")
        return log_path

    def setup_file_handler(self):
        """
        Sets up a rotating file handler for the log file.

        Returns:
            RotatingFileHandler: Configured file handler.
        """
        log_path = self.get_log_file_path()
        handler = logging.handlers.RotatingFileHandler(
            log_path, maxBytes=10 * 1024 * 1024, backupCount=5  # 10MB
        )
        handler.setLevel(logging.DEBUG)  # Capture all levels (DEBUG and above)
        handler.setFormatter(self.detailed_formatter)
        return handler

    def setup_console_handler(self):
        """
        Sets up a console handler for logging.

        Returns:
            StreamHandler: Configured console handler.
        """
        console_handler = logging.StreamHandler()
        console_handler.setLevel(
            logging.INFO
        )  # Only show INFO and above in the console
        console_handler.setFormatter(self.console_formatter)
        return console_handler

    def configure_logging(self):
        """
        Configures logging settings with a single file handler and a console handler.
        """
        try:
            # Get the root logger
            logger = logging.getLogger()
            logger.setLevel(logging.DEBUG)  # Set the base logging level

            # Remove any existing handlers
            for handler in logger.handlers[:]:
                logger.removeHandler(handler)

            # Set up a single file handler for all log levels
            file_handler = self.setup_file_handler()
            console_handler = self.setup_console_handler()

            # Add handlers
            logger.addHandler(file_handler)
            logger.addHandler(console_handler)

            # Prevent duplicate logs
            logger.propagate = False

            print(f"[DEBUG] Logging successfully configured with a single file handler")

        except Exception as e:
            print(f"Failed to configure logging: {e}")
            raise


def configure_logging(app_name=None):
    """
    Main function to configure logging system.

    Args:
        app_name (str, optional): Name of the application.
    """
    logger_setup = EnhancedLoggerSetup(app_name)
    logger_setup.configure_logging()


# Automatically configure logging when this module is imported
if __name__ != "__main__":
    configure_logging()
