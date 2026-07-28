"""Base LLM client (shared infrastructure).

Provides :class:`ResponseCache` and :class:`BaseLLMClient`.  Every VLM
backend (Gemini, NVIDIA, mock) inherits from :class:`BaseLLMClient` and
implements only :meth:`_produce_raw_text`; caching, rate-limiting, call
budget, JSON parsing and schema validation are handled here.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

from llm.gemini_types import GeminiResponseError, MaxCallsExceededError
from llm.rate_limiter import RateLimiter
from utils.hash_utils import short_hash
from utils.json_utils import coerce_json_to_schema, parse_json_lenient, validate_json_schema
from utils.logging_utils import get_logger

logger = get_logger(__name__)


class ResponseCache:
    """Minimal JSON file cache keyed on a caller-supplied cache key."""

    def __init__(self, cache_dir: str):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _path_for(self, cache_key: str) -> Path:
        return self.cache_dir / f"{short_hash(cache_key, 32)}.json"

    def get(self, cache_key: str) -> Optional[dict]:
        path = self._path_for(cache_key)
        if not path.exists():
            return None
        try:
            with open(path, "r", encoding="utf-8") as handle:
                return json.load(handle)
        except (OSError, json.JSONDecodeError) as error:
            logger.warning("Ignoring unreadable cache file %s: %s", path, error)
            return None

    def set(self, cache_key: str, value: dict) -> None:
        path = self._path_for(cache_key)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)


class BaseLLMClient:
    """Shared structured-JSON client behaviour (cache, retry, validation).

    Subclasses implement :meth:`_produce_raw_text`, which returns the raw
    JSON text for a prompt + images.  Everything else runs through here.
    """

    client_kind = "base"

    def __init__(
        self,
        model_name: str,
        cache: Optional[ResponseCache],
        rate_limiter: RateLimiter,
        config,
    ):
        self.model_name = model_name
        self.cache = cache
        self.rate_limiter = rate_limiter
        self.config = config
        self.call_count = 0

    def _produce_raw_text(
        self, prompt: str, schema: dict, image_paths: List[str]
    ) -> str:
        raise NotImplementedError

    def _check_call_budget(self) -> None:
        if self.call_count >= self.config.max_calls_per_run:
            raise MaxCallsExceededError(
                f"Reached max_calls_per_run={self.config.max_calls_per_run}."
            )

    def generate_json(
        self,
        prompt: str,
        schema: dict,
        image_paths: List[str],
        cache_key: str,
    ) -> dict:
        """Generate a JSON object from a prompt and images."""
        cache_enabled = getattr(self.config, "cache_enabled", True)

        if self.cache is not None and cache_enabled:
            cached = self.cache.get(cache_key)
            if cached is not None:
                logger.info("Cache hit for cache_key=%s", cache_key)
                return cached

        self._check_call_budget()
        self.rate_limiter.wait()

        logger.info(
            "%s call #%d model=%s images=%d cache_key=%s",
            self.client_kind,
            self.call_count + 1,
            self.model_name,
            len([p for p in image_paths if p]),
            cache_key,
        )

        raw_text = self._produce_raw_text(prompt, schema, image_paths)
        self.call_count += 1

        try:
            data = parse_json_lenient(raw_text)
        except Exception as error:  # noqa: BLE001
            snippet = (raw_text or "")[:500]
            raise GeminiResponseError(
                f"Model returned invalid JSON: {error}. Raw text: {snippet!r}"
            ) from error

        if not isinstance(data, dict):
            raise GeminiResponseError(
                f"Expected a JSON object, got {type(data).__name__}."
            )

        data = coerce_json_to_schema(data, schema)
        validate_json_schema(data, schema)

        if self.cache is not None and cache_enabled:
            self.cache.set(cache_key, data)

        return data
