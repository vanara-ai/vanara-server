import logging
import sys
from datetime import datetime


class RenderFormatter(logging.Formatter):
    """Custom formatter for Render logs that makes them easily searchable and readable"""

    def format(self, record: logging.LogRecord) -> str:
        # Get the basic message
        message = record.getMessage()

        # Get extra fields directly from record's __dict__
        extra = {
            k: v
            for k, v in record.__dict__.items()
            if k
            not in [
                "args",
                "asctime",
                "created",
                "exc_info",
                "exc_text",
                "filename",
                "funcName",
                "id",
                "levelname",
                "levelno",
                "lineno",
                "module",
                "msecs",
                "message",
                "msg",
                "name",
                "pathname",
                "process",
                "processName",
                "relativeCreated",
                "stack_info",
                "thread",
                "threadName",
                "taskName",
            ]
        }  # Filter out taskName

        # Format timestamp
        timestamp = datetime.utcnow().isoformat()

        # Build the log line
        log_line = (
            f"[{timestamp}] [{record.levelname}] [{record.filename}] [{record.funcName}] [{record.lineno}] {message}"
        )

        # Add extra fields if they exist
        if extra:
            extra_str = " | ".join(f"{k}={v}" for k, v in extra.items())
            log_line += f" | Extra: {extra_str}"

        # Add exception if it exists
        if record.exc_info:
            log_line += f"\nException:\n{self.formatException(record.exc_info)}"

        return log_line


def setup_logger(name: str = "resume_ai") -> logging.Logger:
    """Configure and return a logger instance"""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Remove any existing handlers
    logger.handlers = []

    # Console Handler with custom formatter
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(RenderFormatter())
    logger.addHandler(console_handler)

    return logger


# Create a default logger instance
logger = setup_logger()
