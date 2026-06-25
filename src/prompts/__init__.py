"""Versioned Gemini prompts (Phase 8).

Each prompt module exposes ``PROMPT_NAME``, ``PROMPT_VERSION``,
``OUTPUT_SCHEMA``, ``FEW_SHOT_EXAMPLES`` and ``build_prompt(...)``. All
Gemini outputs are strict JSON; no unstructured free text is allowed.
"""

from __future__ import annotations

from prompts import (
    answerability_prompt,
    layered_scene_prompt,
    question_generation_prompt,
    scenario_prompt,
)
from prompts.base import build_cache_key, select_image_paths

__all__ = [
    "layered_scene_prompt",
    "scenario_prompt",
    "question_generation_prompt",
    "answerability_prompt",
    "build_cache_key",
    "select_image_paths",
]
