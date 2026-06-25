"""Shared types for the Gemini client layer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


class MaxCallsExceededError(RuntimeError):
    """Raised when ``max_calls_per_run`` would be exceeded."""


class GeminiResponseError(RuntimeError):
    """Raised when Gemini returns content that cannot be parsed as JSON."""


@dataclass
class GeminiResult:
    """Structured result of a single Gemini JSON generation call."""

    data: dict
    raw_text: Optional[str]
    cache_key: str
    from_cache: bool
