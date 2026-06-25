"""Logging utilities with API-key redaction.

`setup_logging` configures the root logger once. `register_secret` lets
callers (e.g. the Gemini client) register secret strings that must never
appear in log output; a logging filter masks them automatically.
"""

from __future__ import annotations

import logging
import sys
from typing import Set

_SECRETS: Set[str] = set()
_CONFIGURED = False


def register_secret(value: str | None) -> None:
    """Register a secret value to be redacted from all log records."""
    if value:
        _SECRETS.add(value)


class _RedactionFilter(logging.Filter):
    """Replaces any registered secret substring in log messages."""

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: A003
        if not _SECRETS:
            return True
        message = record.getMessage()
        redacted = message
        for secret in _SECRETS:
            if secret and secret in redacted:
                redacted = redacted.replace(secret, "***REDACTED***")
        if redacted != message:
            record.msg = redacted
            record.args = ()
        return True


def setup_logging(level: int | str = logging.INFO) -> None:
    """Configure the root logger with a console handler exactly once."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    handler.addFilter(_RedactionFilter())

    root = logging.getLogger()
    root.setLevel(level)
    root.addHandler(handler)
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a logger, ensuring logging is configured first."""
    if not _CONFIGURED:
        setup_logging()
    return logging.getLogger(name)
