"""Logging configuration for PlotSmith."""

import logging
import sys
from pathlib import Path
from typing import Optional

# Default log format
DEFAULT_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def configure_logging(
    level: int = logging.INFO,
    format_string: Optional[str] = None,
    log_file: Optional[str | Path] = None,
    structured: bool = False,
) -> None:
    """Configure logging for PlotSmith.

    Args:
        level: Logging level (default: INFO).
        format_string: Custom format string. If None, uses default.
        log_file: Optional path to log file. If None, logs to stderr.
        structured: If True, use JSON structured logging format.
    """
    if format_string is None:
        format_string = DEFAULT_FORMAT

    # Create formatter
    if structured:
        # JSON structured logging (for production)
        import json
        from datetime import datetime

        class JSONFormatter(logging.Formatter):
            def format(self, record: logging.LogRecord) -> str:
                log_entry = {
                    "timestamp": datetime.utcnow().isoformat(),
                    "level": record.levelname,
                    "logger": record.name,
                    "message": record.getMessage(),
                    "module": record.module,
                    "function": record.funcName,
                    "line": record.lineno,
                }
                if record.exc_info:
                    log_entry["exception"] = self.formatException(record.exc_info)
                return json.dumps(log_entry)

        formatter = JSONFormatter()
    else:
        formatter = logging.Formatter(format_string, datefmt=DEFAULT_DATE_FORMAT)

    # Configure root logger
    root_logger = logging.getLogger("plotsmith")
    root_logger.setLevel(level)
    root_logger.handlers.clear()

    # Console handler (stderr)
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File handler (if specified)
    if log_file:
        file_path = Path(log_file)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(file_path)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    """Get a logger for a specific module.

    Args:
        name: Logger name (typically __name__).

    Returns:
        Configured logger instance.
    """
    return logging.getLogger(f"plotsmith.{name}")


# Configure default logging on import
configure_logging()

