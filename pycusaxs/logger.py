"""
Logging configuration for pyCuSaxs.

This module provides centralized logging configuration for the pyCuSaxs package.
It supports both console and file logging with configurable levels.
"""

import logging
import sys
from pathlib import Path
from typing import Optional


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for different log levels."""

    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
    }
    RESET = '\033[0m'

    def format(self, record):
        """Format log record with colors if terminal supports it."""
        if sys.stderr.isatty():
            levelname = record.levelname
            if levelname in self.COLORS:
                record.levelname = f"{self.COLORS[levelname]}{levelname}{self.RESET}"
                record.msg = f"{self.COLORS[levelname]}{record.msg}{self.RESET}"
        return super().format(record)


def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[Path] = None,
    verbose: bool = False
) -> logging.Logger:
    """
    Configure logging for pyCuSaxs.

    Args:
        level: Logging level (default: INFO)
        log_file: Optional path to log file
        verbose: If True, set level to DEBUG

    Returns:
        Configured logger instance
    """
    if verbose:
        level = logging.DEBUG

    # Create logger
    logger = logging.getLogger('pycusaxs')
    logger.setLevel(level)

    # Remove existing handlers
    logger.handlers.clear()

    # Console handler with colored output
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(level)
    console_formatter = ColoredFormatter(
        '%(levelname)s - %(message)s'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # File handler (if specified)
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)  # Always log everything to file
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Get logger instance for a specific module.

    Args:
        name: Module name (default: 'pycusaxs')

    Returns:
        Logger instance
    """
    if name:
        return logging.getLogger(f'pycusaxs.{name}')
    return logging.getLogger('pycusaxs')
