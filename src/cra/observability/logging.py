"""Structured JSON logging with request-ID propagation and PII redaction.

Every log line is emitted as a single JSON object. A ``request_id`` is carried
through a :class:`contextvars.ContextVar` so all logs within one request are
correlated without threading the ID through every call. A logging filter
redacts PII from messages before they are serialized.
"""

from __future__ import annotations

import logging
from contextvars import ContextVar

from pythonjsonlogger import jsonlogger

from cra.observability.pii import redact

_request_id_var: ContextVar[str] = ContextVar("request_id", default="-")

_CONFIGURED = False


def set_request_id(request_id: str) -> None:
    _request_id_var.set(request_id)


def get_request_id() -> str:
    return _request_id_var.get()


class _RequestIdFilter(logging.Filter):
    """Attach the current request_id and redact the rendered message."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = _request_id_var.get()
        if isinstance(record.msg, str):
            record.msg = redact(record.msg)
        return True


def configure_logging(level: str = "INFO") -> None:
    """Install the JSON handler on the root logger (idempotent)."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    handler = logging.StreamHandler()
    formatter = jsonlogger.JsonFormatter(
        "%(asctime)s %(levelname)s %(name)s %(request_id)s %(message)s",
        rename_fields={"asctime": "timestamp", "levelname": "level"},
    )
    handler.setFormatter(formatter)
    handler.addFilter(_RequestIdFilter())

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level.upper())
    _CONFIGURED = True


def get_logger(name: str) -> logging.LoggerAdapter:
    """Return a logger. ``configure_logging`` must have been called first."""
    return logging.LoggerAdapter(logging.getLogger(name), extra={})
