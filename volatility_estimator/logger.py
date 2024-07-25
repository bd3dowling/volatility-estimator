import json
import logging
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path


class JsonFormatter(logging.Formatter):
    def format(self, record):
        log_record = {
            "time": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "message": record.getMessage(),
            "pathname": record.pathname,
            "lineno": record.lineno,
            "funcName": record.funcName,
            "stack_info": self.formatStack(record.stack_info) if record.stack_info else None,
            "exc_info": self.formatException(record.exc_info) if record.exc_info else None,
        }
        return json.dumps(log_record)


def get_logger():
    logger = logging.getLogger("app")

    # Ensure don't re-add handler from multiple calls, preventing double logging
    if not logger.handlers:
        handler = TimedRotatingFileHandler(
            Path("logs") / "app_log.json",
            when="midnight",
            interval=1,
            backupCount=0,
        )
        handler.setFormatter(JsonFormatter())
        logger.setLevel(logging.INFO)
        logger.addHandler(handler)

    return logger
