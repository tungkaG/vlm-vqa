"""JSON and JSONL storage helpers.

All functions accept either plain dictionaries or Pydantic models
(anything exposing ``model_dump``). Files are written as UTF-8 with
``ensure_ascii=False`` so non-ASCII text stays readable.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, List


def _to_jsonable(record: Any) -> Any:
    """Convert Pydantic models to plain dictionaries; pass dicts through."""
    if hasattr(record, "model_dump"):
        return record.model_dump()
    return record


def ensure_dir(path: str | Path) -> None:
    """Create a directory (and parents) if it does not already exist."""
    Path(path).mkdir(parents=True, exist_ok=True)


def save_json(obj: Any, path: str | Path) -> None:
    """Serialize a single object to a JSON file."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(_to_jsonable(obj), handle, ensure_ascii=False, indent=2)


def load_json(path: str | Path) -> Any:
    """Load a single object from a JSON file."""
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def save_jsonl(records: Iterable[Any], path: str | Path) -> None:
    """Write an iterable of records as JSON Lines (overwrites)."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(_to_jsonable(record), ensure_ascii=False))
            handle.write("\n")


def append_jsonl(record: Any, path: str | Path) -> None:
    """Append a single record to a JSON Lines file."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(_to_jsonable(record), ensure_ascii=False))
        handle.write("\n")


def load_jsonl(path: str | Path) -> List[dict]:
    """Load all records from a JSON Lines file."""
    records: List[dict] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_jsonl_if_exists(path: str | Path) -> List[dict]:
    """Load a JSON Lines file, returning an empty list if it is absent."""
    if not Path(path).exists():
        return []
    return load_jsonl(path)
