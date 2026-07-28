"""NVIDIA NIM client (OpenAI-compatible endpoint).

Uses the `openai` SDK pointed at ``https://integrate.api.nvidia.com/v1``.
Images are sent as inline base64 data-URL parts in the ``image_url``
content block.  Reasoning is explicitly disabled (``enable_thinking=False``)
so the model returns clean text; any stray ``<think>…</think>`` prefix is
stripped defensively before JSON parsing.
"""

from __future__ import annotations

from typing import List, Optional

from openai import OpenAI

from llm.base_client import BaseLLMClient, ResponseCache
from llm.rate_limiter import RateLimiter
from llm.retry import call_with_retry
from utils.image_utils import mime_type_for
from utils.logging_utils import get_logger, register_secret

logger = get_logger(__name__)


def _build_image_data_urls(image_paths: List[str], max_images: int) -> List[dict]:
    """Return OpenAI-style image_url content blocks (base64 inline)."""
    import base64

    selected = [p for p in image_paths if p]
    if len(selected) > max_images:
        logger.warning(
            "Got %d images but max_images_per_request=%d; truncating.",
            len(selected),
            max_images,
        )
        selected = selected[:max_images]

    parts = []
    for path in selected:
        with open(path, "rb") as handle:
            data = base64.b64encode(handle.read()).decode("ascii")
        mime = mime_type_for(path)
        parts.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{data}"},
            }
        )
    return parts


def _strip_thinking(text: str) -> str:
    """Remove a leading ``<think>…</think>`` block if present."""
    if text and "</think>" in text:
        idx = text.index("</think>")
        return text[idx + len("</think>"):].lstrip()
    return text


class NvidiaClient(BaseLLMClient):
    """LLM client for the NVIDIA NIM OpenAI-compatible API."""

    client_kind = "nvidia"

    def __init__(
        self,
        api_key: str,
        model_name: str,
        base_url: str,
        cache: Optional[ResponseCache],
        rate_limiter: RateLimiter,
        config,
    ):
        if not api_key:
            raise RuntimeError("NVIDIA API key is empty.")
        register_secret(api_key)
        super().__init__(model_name, cache, rate_limiter, config)
        self._client = OpenAI(api_key=api_key, base_url=base_url)

    def _produce_raw_text(
        self, prompt: str, schema: dict, image_paths: List[str]
    ) -> str:
        max_images = getattr(self.config, "max_images_per_request", 6)
        image_parts = _build_image_data_urls(image_paths, max_images)

        content: List[dict] = [{"type": "text", "text": prompt}] + image_parts

        max_tokens = getattr(self.config, "max_output_tokens", 4096)
        timeout = getattr(self.config, "request_timeout_seconds", 120)

        def _api_call():
            return self._client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": content}],
                temperature=0.2,
                max_tokens=max_tokens,
                timeout=timeout,
                extra_body={
                    "top_k": 1,
                    "chat_template_kwargs": {"enable_thinking": False},
                },
            )

        response = call_with_retry(
            _api_call,
            max_retries=self.config.max_retries,
            backoff_seconds=self.config.retry_backoff_seconds,
        )
        raw = response.choices[0].message.content or ""
        return _strip_thinking(raw)
