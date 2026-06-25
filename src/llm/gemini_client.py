"""Gemini client layer (Phase 5).

Every VLM call in the pipeline goes through a client exposing
``generate_json(prompt, schema, image_paths, cache_key)``. The base class
:class:`BaseGeminiClient` implements all the surrounding behaviour
(caching, rate limiting, call budget, JSON parsing and schema
validation). Only the single method :meth:`_produce_raw_text` differs
between the real Gemini call and the mock, so the call path is identical.

The dedicated prompt-cache module (README Phase 6) is intentionally not
implemented; a small file cache keyed on the caller-supplied
``cache_key`` provides the caching required by Phase 5.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

from google import genai
from google.genai import types

from config import AppConfig, get_gemini_api_key, resolve_model_name
from llm.gemini_types import GeminiResponseError, MaxCallsExceededError
from llm.image_payload import build_image_parts
from llm.rate_limiter import RateLimiter
from llm.retry import call_with_retry
from utils.hash_utils import short_hash
from utils.json_utils import parse_json_lenient, validate_json_schema
from utils.logging_utils import get_logger, register_secret

logger = get_logger(__name__)


class ResponseCache:
    """Minimal JSON file cache keyed on a caller-supplied cache key."""

    def __init__(self, cache_dir: str):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _path_for(self, cache_key: str) -> Path:
        # Hash the key so arbitrary strings map to safe file names.
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


class BaseGeminiClient:
    """Shared structured-JSON client behaviour (cache, retry, validation).

    Subclasses implement :meth:`_produce_raw_text`, which returns the raw
    JSON text for a prompt + images. Everything else — caching, the call
    budget, rate limiting, parsing and schema validation — is identical
    for the real and mock clients.
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
        """Raise if the per-run paid-call budget would be exceeded."""
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
        """Generate a JSON object from a prompt and images.

        Returns the parsed (and schema-validated) JSON dict. Reads from
        cache when available; otherwise produces one response.
        """
        cache_enabled = getattr(self.config, "cache_enabled", True)

        # 1. Cache lookup.
        if self.cache is not None and cache_enabled:
            cached = self.cache.get(cache_key)
            if cached is not None:
                logger.info("Cache hit for cache_key=%s", cache_key)
                return cached

        # 2. Enforce the per-run call budget *before* spending a call.
        self._check_call_budget()

        # 3. Respect the rate limit.
        self.rate_limiter.wait()

        logger.info(
            "%s call #%d model=%s images=%d cache_key=%s",
            self.client_kind,
            self.call_count + 1,
            self.model_name,
            len([p for p in image_paths if p]),
            cache_key,
        )

        # 4. Produce raw JSON text (real API call or mock).
        raw_text = self._produce_raw_text(prompt, schema, image_paths)
        self.call_count += 1

        # 5. Parse JSON. Invalid JSON surfaces clearly (not retried away).
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

        # 6. Validate against the provided schema.
        validate_json_schema(data, schema)

        # 7. Cache the validated response.
        if self.cache is not None and cache_enabled:
            self.cache.set(cache_key, data)

        return data


class GeminiClient(BaseGeminiClient):
    """Real Gemini client using the google-genai Developer API."""

    client_kind = "gemini"

    def __init__(
        self,
        api_key: str,
        model_name: str,
        cache: Optional[ResponseCache],
        rate_limiter: RateLimiter,
        config,
    ):
        if not api_key:
            raise RuntimeError("Gemini API key is empty.")
        # Ensure the key can never leak into logs.
        register_secret(api_key)
        super().__init__(model_name, cache, rate_limiter, config)
        self._client = genai.Client(api_key=api_key)

    def _produce_raw_text(
        self, prompt: str, schema: dict, image_paths: List[str]
    ) -> str:
        # Build request contents (prompt text + inline images).
        image_parts = build_image_parts(
            image_paths, self.config.max_images_per_request
        )
        contents = [prompt, *image_parts]
        gen_config = types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.2,
        )

        def _api_call():
            return self._client.models.generate_content(
                model=self.model_name,
                contents=contents,
                config=gen_config,
            )

        response = call_with_retry(
            _api_call,
            max_retries=self.config.max_retries,
            backoff_seconds=self.config.retry_backoff_seconds,
        )
        return getattr(response, "text", None)


def build_gemini_client(config: AppConfig):
    """Construct a Gemini client (real or mock) from an :class:`AppConfig`.

    Mock mode is selected when ``gemini.mock_mode`` is true or the
    ``GEMINI_MOCK`` environment variable is set, and requires no API key.
    """
    import os

    model_name = resolve_model_name(config)
    cache = (
        ResponseCache(config.gemini.cache_dir)
        if config.gemini.cache_enabled
        else None
    )
    rate_limiter = RateLimiter(config.gemini.min_seconds_between_calls)

    mock_mode = bool(getattr(config.gemini, "mock_mode", False)) or bool(
        os.environ.get("GEMINI_MOCK")
    )

    if mock_mode:
        from llm.mock_gemini_client import MockGeminiClient

        logger.warning(
            "MOCK MODE active: no real Gemini calls will be made. "
            "Set gemini.mock_mode=false (and unset GEMINI_MOCK) to use the API."
        )
        return MockGeminiClient(
            model_name=model_name,
            cache=cache,
            # Mock calls hit no network, so no inter-call delay is needed.
            rate_limiter=RateLimiter(0.0),
            config=config.gemini,
        )

    api_key = get_gemini_api_key(config)
    return GeminiClient(
        api_key=api_key,
        model_name=model_name,
        cache=cache,
        rate_limiter=rate_limiter,
        config=config.gemini,
    )
