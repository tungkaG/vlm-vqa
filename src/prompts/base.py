"""Shared prompt helpers (Phase 8).

Provides :func:`build_cache_key`, used by every annotator stage to derive
a stable cache key from the sample, prompt identity, model and the exact
set of preview images (their file names already encode a content hash).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from utils.hash_utils import short_hash


def select_image_paths(
    preview_paths: Dict[str, Optional[str]], max_images: int
) -> List[str]:
    """Return up to ``max_images`` existing preview paths (missing ones dropped).

    Preserves the camera ordering of ``preview_paths`` (front cameras first).
    """
    paths = [path for path in preview_paths.values() if path]
    if max_images > 0:
        paths = paths[:max_images]
    return paths


def build_cache_key(
    *,
    sample_id: str,
    prompt_name: str,
    prompt_version: str,
    model_name: str,
    image_paths: List[str],
    schema_version: str = "1",
    question: Optional[str] = None,
    context: Optional[str] = None,
) -> str:
    """Build a deterministic cache key for a Gemini call.

    The key combines the sample id, prompt identity/version, model name,
    schema version, the sorted preview file names (which already encode a
    content hash), an optional upstream-context string (hashed, so a stage
    re-runs when an earlier stage's output changes) and an optional
    question string.
    """
    basenames = sorted(Path(path).name for path in image_paths if path)
    parts = [
        sample_id,
        prompt_name,
        prompt_version,
        model_name,
        f"schema={schema_version}",
        "imgs=" + ",".join(basenames),
    ]
    if context:
        parts.append("ctx=" + short_hash(context, 16))
    if question:
        parts.append("q=" + question.strip())
    return "::".join(parts)
