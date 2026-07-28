"""Unit tests for the NVIDIA client and LLM factory.

No real API calls are made: the openai client's ``create`` method is
patched with ``unittest.mock``.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from config import AppConfig, DatasetConfig, LLMConfig
from llm.base_client import ResponseCache
from llm.nvidia_client import NvidiaClient, _strip_thinking, _build_image_data_urls
from llm.rate_limiter import RateLimiter


def _nvidia_config(**overrides) -> LLMConfig:
    defaults = dict(
        provider="nvidia",
        base_url="https://integrate.api.nvidia.com/v1",
        api_key_env="NVIDIA_API_KEY",
        model_name_env="LLM_MODEL_NAME",
        default_model_name="nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",
        max_calls_per_run=10,
        request_timeout_seconds=30,
        max_retries=1,
        retry_backoff_seconds=0.0,
        min_seconds_between_calls=0.0,
        cache_enabled=False,
        cache_dir="outputs/cache/llm",
        max_output_tokens=256,
        use_structured_output=False,
        image_input_mode="inline",
        max_images_per_request=2,
        max_image_side_pixels=640,
        mock_mode=False,
    )
    defaults.update(overrides)
    return LLMConfig(**defaults)


def _make_client(config: LLMConfig) -> NvidiaClient:
    with patch("llm.nvidia_client.OpenAI"):
        client = NvidiaClient(
            api_key="test-key",
            model_name="nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",
            base_url="https://integrate.api.nvidia.com/v1",
            cache=None,
            rate_limiter=RateLimiter(0.0),
            config=config,
        )
    return client


# ---------------------------------------------------------------------------
# _strip_thinking
# ---------------------------------------------------------------------------

def test_strip_thinking_with_tag():
    raw = "<think>step by step</think>\n\n{\"key\": \"value\"}"
    assert _strip_thinking(raw) == "{\"key\": \"value\"}"


def test_strip_thinking_without_tag():
    raw = "{\"key\": \"value\"}"
    assert _strip_thinking(raw) == raw


def test_strip_thinking_empty():
    assert _strip_thinking("") == ""


# ---------------------------------------------------------------------------
# _build_image_data_urls
# ---------------------------------------------------------------------------

def test_build_image_data_urls_truncates(tmp_path):
    # Create 3 tiny JPEG files.
    imgs = []
    for i in range(3):
        p = tmp_path / f"img{i}.jpg"
        p.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 10)  # minimal JPEG header
        imgs.append(str(p))
    parts = _build_image_data_urls(imgs, max_images=2)
    assert len(parts) == 2
    for part in parts:
        assert part["type"] == "image_url"
        assert part["image_url"]["url"].startswith("data:image/jpeg;base64,")


def test_build_image_data_urls_empty():
    assert _build_image_data_urls([], max_images=6) == []


# ---------------------------------------------------------------------------
# NvidiaClient._produce_raw_text
# ---------------------------------------------------------------------------

def _mock_response(content: str):
    choice = SimpleNamespace(message=SimpleNamespace(content=content))
    return SimpleNamespace(choices=[choice])


def test_nvidia_client_produce_raw_text(tmp_path):
    config = _nvidia_config()
    client = _make_client(config)

    # Patch the underlying openai create call.
    client._client.chat.completions.create = MagicMock(
        return_value=_mock_response('{"scene_summary": "test", "main_objects": []}')
    )

    schema = {
        "type": "object",
        "required": ["scene_summary", "main_objects"],
        "properties": {
            "scene_summary": {"type": "string"},
            "main_objects": {"type": "array", "items": {"type": "string"}},
        },
    }
    result = client.generate_json(
        prompt="Describe the scene.",
        schema=schema,
        image_paths=[],
        cache_key="test::key",
    )
    assert result["scene_summary"] == "test"
    assert client.call_count == 1

    # Verify enable_thinking=False is passed.
    call_kwargs = client._client.chat.completions.create.call_args
    extra = call_kwargs.kwargs.get("extra_body") or call_kwargs[1].get("extra_body", {})
    assert extra["chat_template_kwargs"]["enable_thinking"] is False


def test_nvidia_client_strips_thinking_prefix(tmp_path):
    config = _nvidia_config()
    client = _make_client(config)

    client._client.chat.completions.create = MagicMock(
        return_value=_mock_response(
            '<think>reasoning here</think>\n{"scene_summary": "ok", "main_objects": []}'
        )
    )
    schema = {
        "type": "object",
        "required": ["scene_summary", "main_objects"],
        "properties": {
            "scene_summary": {"type": "string"},
            "main_objects": {"type": "array", "items": {"type": "string"}},
        },
    }
    result = client.generate_json("p", schema, [], "key2")
    assert result["scene_summary"] == "ok"


# ---------------------------------------------------------------------------
# Factory dispatch
# ---------------------------------------------------------------------------

def test_factory_dispatches_mock(monkeypatch):
    monkeypatch.setenv("LLM_MOCK", "1")
    from config import AppConfig, DatasetConfig
    from llm.factory import build_llm_client
    from llm.mock_gemini_client import MockGeminiClient

    config = AppConfig(dataset=DatasetConfig(dataroot="C:/none", version="v1.0-mini"))
    client = build_llm_client(config)
    assert isinstance(client, MockGeminiClient)


def test_factory_dispatches_nvidia(monkeypatch):
    monkeypatch.delenv("LLM_MOCK", raising=False)
    monkeypatch.delenv("GEMINI_MOCK", raising=False)
    monkeypatch.setenv("NVIDIA_API_KEY", "fake-key")

    from config import AppConfig, DatasetConfig
    from llm.factory import build_llm_client
    from llm.nvidia_client import NvidiaClient

    with patch("llm.nvidia_client.OpenAI"):
        config = AppConfig(dataset=DatasetConfig(dataroot="C:/none", version="v1.0-mini"))
        client = build_llm_client(config)
    assert isinstance(client, NvidiaClient)


def test_factory_raises_on_unknown_provider():
    from config import AppConfig, DatasetConfig, LLMConfig
    from pydantic import Field
    from llm.factory import build_llm_client

    config = AppConfig(
        dataset=DatasetConfig(dataroot="C:/none", version="v1.0-mini"),
        llm=LLMConfig(provider="unknown_xyz"),
    )
    with pytest.raises(ValueError, match="unknown_xyz"):
        build_llm_client(config)
