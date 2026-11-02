"""Compact JSON event logger for application lifecycle and HTTP activity."""

import json
import logging
import time
from typing import Any, Optional


class StructuredLogger:
    def __init__(self, name: str = "rice_analysis_events") -> None:
        self._logger = logging.getLogger(name)
        self._logger.propagate = False

    @property
    def logger(self) -> logging.Logger:
        return self._logger

    def _emit(self, level: int, event: str, request_id: Optional[str] = None, **fields: Any) -> None:
        now = time.time()
        timestamp = time.strftime("%Y-%m-%dT%H:%M:%S.", time.gmtime(now)) + f"{int(now * 1000) % 1000:03d}Z"
        record = {
            "timestamp": timestamp,
            "level": logging.getLevelName(level),
            "event": event,
            "request_id": request_id,
            **fields,
        }
        self._logger.log(
            level,
            json.dumps({key: value for key, value in record.items() if value is not None}, default=str),
        )

    def info(self, event: str, request_id: Optional[str] = None, **fields: Any) -> None:
        self._emit(logging.INFO, event, request_id, **fields)

    def warning(self, event: str, request_id: Optional[str] = None, **fields: Any) -> None:
        self._emit(logging.WARNING, event, request_id, **fields)

    def error(self, event: str, request_id: Optional[str] = None, **fields: Any) -> None:
        self._emit(logging.ERROR, event, request_id, **fields)


structured_logger = StructuredLogger()
