"""Tests for local LLM-support logic (Phases 5 & 7).

No real Gemini calls are made; only the dependency-free helpers
(JSON parsing, minimal schema validation, rate limiting, retry) are
exercised.
"""

import time

import pytest

from llm.gemini_client import ResponseCache
from llm.rate_limiter import RateLimiter
from llm.retry import call_with_retry
from utils.json_utils import parse_json_lenient, validate_json_schema


def test_parse_plain_json():
    assert parse_json_lenient('{"a": 1}') == {"a": 1}


def test_parse_json_with_code_fence():
    text = '```json\n{"a": 1, "b": [2, 3]}\n```'
    assert parse_json_lenient(text) == {"a": 1, "b": [2, 3]}


def test_parse_json_embedded_in_prose():
    text = 'Here is the result:\n{"ok": true}\nThanks!'
    assert parse_json_lenient(text) == {"ok": True}


def test_validate_schema_ok():
    schema = {
        "type": "object",
        "required": ["name", "items"],
        "properties": {
            "name": {"type": "string"},
            "items": {"type": "array", "items": {"type": "string"}},
        },
    }
    validate_json_schema({"name": "x", "items": ["a", "b"]}, schema)  # no error


def test_validate_schema_missing_required():
    schema = {"type": "object", "required": ["name"], "properties": {}}
    with pytest.raises(ValueError):
        validate_json_schema({}, schema)


def test_validate_schema_wrong_type():
    schema = {"type": "object", "properties": {"n": {"type": "integer"}}}
    with pytest.raises(ValueError):
        validate_json_schema({"n": "not-an-int"}, schema)


def test_validate_schema_rejects_bool_as_number():
    schema = {"type": "object", "properties": {"n": {"type": "number"}}}
    with pytest.raises(ValueError):
        validate_json_schema({"n": True}, schema)


def test_rate_limiter_spaces_calls():
    limiter = RateLimiter(min_seconds_between_calls=0.2)
    limiter.wait()  # first call returns immediately
    start = time.monotonic()
    limiter.wait()  # second call must wait ~0.2s
    assert time.monotonic() - start >= 0.18


def test_retry_succeeds_after_failures():
    attempts = {"count": 0}

    def flaky():
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise RuntimeError("transient")
        return "ok"

    assert call_with_retry(flaky, max_retries=5, backoff_seconds=0) == "ok"
    assert attempts["count"] == 3


def test_retry_reraises_after_exhaustion():
    def always_fails():
        raise RuntimeError("permanent")

    with pytest.raises(RuntimeError, match="permanent"):
        call_with_retry(always_fails, max_retries=2, backoff_seconds=0)


def test_response_cache_round_trip(tmp_path):
    cache = ResponseCache(str(tmp_path / "cache"))
    assert cache.get("key-1") is None
    cache.set("key-1", {"value": 42})
    assert cache.get("key-1") == {"value": 42}
