"""Retry helper for transient failures (Phase 7).

``call_with_retry`` retries a callable on exception with linear backoff.
The final failure is logged clearly and re-raised — it is never hidden.
Callers should keep JSON parsing/validation *outside* the retried
callable when they want invalid-JSON errors to surface immediately.
"""

from __future__ import annotations

import time
from typing import Callable, TypeVar

from utils.logging_utils import get_logger

logger = get_logger(__name__)

T = TypeVar("T")


def call_with_retry(
    fn: Callable[[], T],
    max_retries: int,
    backoff_seconds: float,
) -> T:
    """Call ``fn`` retrying up to ``max_retries`` times on exception."""
    attempt = 0
    while True:
        try:
            return fn()
        except Exception as error:  # noqa: BLE001 - retry any transient error
            attempt += 1
            if attempt > max_retries:
                logger.error(
                    "Call failed after %d retr%s: %s",
                    max_retries,
                    "y" if max_retries == 1 else "ies",
                    error,
                )
                raise
            sleep_for = backoff_seconds * attempt
            logger.warning(
                "Call failed (attempt %d/%d): %s. Retrying in %.1fs",
                attempt,
                max_retries,
                error,
                sleep_for,
            )
            time.sleep(sleep_for)
