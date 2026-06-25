"""Simple wall-clock rate limiter (Phase 7).

Spaces successive calls by at least ``min_seconds_between_calls`` using a
monotonic clock so it is unaffected by system-clock changes.
"""

from __future__ import annotations

import time

from utils.logging_utils import get_logger

logger = get_logger(__name__)


class RateLimiter:
    def __init__(self, min_seconds_between_calls: float):
        self.min_seconds = max(0.0, float(min_seconds_between_calls))
        self._last_call: float | None = None

    def wait(self) -> None:
        """Block until at least ``min_seconds`` have passed since last call."""
        if self.min_seconds <= 0:
            self._last_call = time.monotonic()
            return

        now = time.monotonic()
        if self._last_call is not None:
            remaining = self.min_seconds - (now - self._last_call)
            if remaining > 0:
                logger.debug("Rate limiting: sleeping %.2fs", remaining)
                time.sleep(remaining)
        self._last_call = time.monotonic()
