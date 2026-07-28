"""Gemini client (google-genai SDK backend).

The base class and shared infrastructure live in :mod:`llm.base_client`.
The factory that selects between providers is in :mod:`llm.factory`.
This module is kept for the Gemini-specific :class:`GeminiClient` and the
legacy ``build_gemini_client`` alias (routes to :func:`llm.factory.build_llm_client`).
"""

from __future__ import annotations

from typing import List, Optional

from google import genai
from google.genai import types

from config import AppConfig, get_llm_api_key, resolve_model_name
from llm.base_client import BaseLLMClient, ResponseCache
from llm.image_payload import build_image_parts
from llm.rate_limiter import RateLimiter
from llm.retry import call_with_retry
from utils.logging_utils import get_logger, register_secret

logger = get_logger(__name__)

# Re-export so existing ``from llm.gemini_client import ResponseCache`` keeps working.
__all__ = ["GeminiClient", "ResponseCache", "build_gemini_client"]


class GeminiClient(BaseLLMClient):
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
    """Legacy alias — delegates to :func:`llm.factory.build_llm_client`."""
    from llm.factory import build_llm_client

    return build_llm_client(config)
