"""Gemini-based auto-annotation stages (Phases 9-13)."""

from __future__ import annotations

from annotator.answerability_classifier import classify_answerability_with_gemini
from annotator.candidate_builder import build_candidate_records
from annotator.layered_descriptor import describe_scene_with_gemini
from annotator.question_generator import generate_questions_with_gemini
from annotator.scenario_classifier import classify_scenario_with_gemini

__all__ = [
    "describe_scene_with_gemini",
    "classify_scenario_with_gemini",
    "generate_questions_with_gemini",
    "classify_answerability_with_gemini",
    "build_candidate_records",
]
