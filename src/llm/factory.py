"""LLM client factory.

Dispatches on ``config.llm.provider`` in {nvidia, gemini, mock}.
Mock mode is also activated by the ``LLM_MOCK`` or ``GEMINI_MOCK``
environment variables so the mock path keeps working without code changes.
"""

from __future__ import annotations

import os

from config import AppConfig, resolve_model_name
from llm.base_client import ResponseCache
from llm.rate_limiter import RateLimiter
from utils.logging_utils import get_logger

logger = get_logger(__name__)


def build_llm_client(config: AppConfig):
    """Construct an LLM client (nvidia / gemini / mock) from :class:`AppConfig`."""
    model_name = resolve_model_name(config)
    cache = (
        ResponseCache(config.llm.cache_dir)
        if config.llm.cache_enabled
        else None
    )
    rate_limiter = RateLimiter(config.llm.min_seconds_between_calls)

    # Mock mode: env var overrides config flag, both trigger mock.
    mock_mode = bool(getattr(config.llm, "mock_mode", False)) or bool(
        os.environ.get("LLM_MOCK") or os.environ.get("GEMINI_MOCK")
    )
    if mock_mode:
        from llm.mock_gemini_client import MockGeminiClient

        logger.warning(
            "MOCK MODE active: no real LLM calls will be made. "
            "Unset LLM_MOCK / GEMINI_MOCK and set provider to use the real API."
        )
        return MockGeminiClient(
            model_name=model_name,
            cache=cache,
            rate_limiter=RateLimiter(0.0),
            config=config.llm,
        )

    provider = getattr(config.llm, "provider", "nvidia").lower()

    if provider == "nvidia":
        from config import load_required_env
        from llm.nvidia_client import NvidiaClient

        api_key = load_required_env(config.llm.api_key_env)
        return NvidiaClient(
            api_key=api_key,
            model_name=model_name,
            base_url=config.llm.base_url,
            cache=cache,
            rate_limiter=rate_limiter,
            config=config.llm,
        )

    if provider == "gemini":
        from config import get_llm_api_key
        from llm.gemini_client import GeminiClient

        api_key = get_llm_api_key(config)
        return GeminiClient(
            api_key=api_key,
            model_name=model_name,
            cache=cache,
            rate_limiter=rate_limiter,
            config=config.llm,
        )

    raise ValueError(
        f"Unknown llm.provider={provider!r}. "
        "Valid values: 'nvidia', 'gemini', 'mock'."
    )


# Back-compat alias used by existing imports in main.py / tests.
build_gemini_client = build_llm_client
