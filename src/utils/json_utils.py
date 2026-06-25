"""JSON parsing and lightweight schema-validation helpers.

`parse_json_lenient` tolerates Markdown code fences that language models
sometimes wrap around JSON. `validate_json_schema` implements a tiny,
dependency-free subset of JSON Schema (type + required) so the Gemini
client can validate responses without pulling in `jsonschema`.
"""

from __future__ import annotations

import json
from typing import Any, Dict

_JSON_TYPES = {
    "object": dict,
    "array": list,
    "string": str,
    "boolean": bool,
    "number": (int, float),
    "integer": int,
    "null": type(None),
}


def parse_json_lenient(text: str) -> Any:
    """Parse JSON, tolerating ```json fences and surrounding prose."""
    if text is None:
        raise ValueError("Cannot parse JSON from None.")

    candidate = text.strip()

    # Strip a Markdown code fence if present.
    if candidate.startswith("```"):
        lines = candidate.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        candidate = "\n".join(lines).strip()

    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        # Fall back to the first balanced {...} or [...] block.
        block = _extract_first_json_block(candidate)
        if block is None:
            raise
        return json.loads(block)


def _extract_first_json_block(text: str) -> str | None:
    start_chars = {"{": "}", "[": "]"}
    for index, char in enumerate(text):
        if char in start_chars:
            closing = start_chars[char]
            depth = 0
            for end in range(index, len(text)):
                if text[end] == char:
                    depth += 1
                elif text[end] == closing:
                    depth -= 1
                    if depth == 0:
                        return text[index : end + 1]
            return None
    return None


def validate_json_schema(data: Any, schema: Dict[str, Any] | None) -> None:
    """Validate `data` against a minimal JSON-Schema subset.

    Supports ``type``, ``required`` and nested ``properties``/``items``.
    Raises ``ValueError`` with a clear message on the first violation.
    """
    if not schema:
        return
    _validate_node(data, schema, path="$")


def _validate_node(data: Any, schema: Dict[str, Any], path: str) -> None:
    expected_type = schema.get("type")
    if expected_type:
        py_type = _JSON_TYPES.get(expected_type)
        if py_type is None:
            raise ValueError(f"Unknown schema type '{expected_type}' at {path}.")
        # bool is a subclass of int; guard against it for number/integer.
        if expected_type in ("number", "integer") and isinstance(data, bool):
            raise ValueError(f"Expected {expected_type} at {path}, got boolean.")
        if not isinstance(data, py_type):
            raise ValueError(
                f"Expected {expected_type} at {path}, got {type(data).__name__}."
            )

    allowed = schema.get("enum")
    if allowed is not None and data not in allowed:
        raise ValueError(f"Value {data!r} at {path} is not one of {allowed}.")

    if expected_type == "object" or "properties" in schema:
        required = schema.get("required", [])
        for key in required:
            if key not in data:
                raise ValueError(f"Missing required key '{key}' at {path}.")
        for key, subschema in schema.get("properties", {}).items():
            if key in data:
                _validate_node(data[key], subschema, f"{path}.{key}")

    if expected_type == "array" and "items" in schema:
        for i, item in enumerate(data):
            _validate_node(item, schema["items"], f"{path}[{i}]")
