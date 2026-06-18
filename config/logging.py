import contextvars
import json
import logging
import sys
from datetime import datetime
from typing import Any, Dict
from config.settings import settings

# Context variable to hold the unique request ID for the lifecycle of a request
request_id_context: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "request_id", default=None
)


class JSONFormatter(logging.Formatter):
    """Custom formatter to output log messages in JSON format."""

    def format(self, record: logging.LogRecord) -> str:
        # Get request_id from context
        request_id = request_id_context.get()

        log_data: Dict[str, Any] = {
            "timestamp": datetime.utcfromtimestamp(record.created).isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": request_id,
        }

        # Include exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        # Include stack trace if requested
        if record.stack_info:
            log_data["stack_info"] = self.formatStack(record.stack_info)

        # Add extra fields if passed
        for key, val in record.__dict__.items():
            if key not in {
                "args",
                "asctime",
                "created",
                "exc_info",
                "exc_text",
                "filename",
                "funcName",
                "levelname",
                "levelno",
                "lineno",
                "module",
                "msecs",
                "msg",
                "name",
                "pathname",
                "process",
                "processName",
                "relativeCreated",
                "stack_info",
                "thread",
                "threadName",
            }:
                log_data[key] = val

        return json.dumps(log_data)


def setup_logging() -> None:
    """Configures the root logger to output JSON logs to stdout."""
    root_logger = logging.getLogger()
    log_level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)
    root_logger.setLevel(log_level)

    # Remove existing handlers to avoid duplicates
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Set up stdout handler with JSONFormatter
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JSONFormatter())
    root_logger.addHandler(handler)
