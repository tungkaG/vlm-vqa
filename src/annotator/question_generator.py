"""Question generation with Gemini (Phase 11)."""

from __future__ import annotations

import json
from typing import List

from prompts import question_generation_prompt
from prompts.base import build_cache_key, select_image_paths
from schema import (
    CandidateQuestion,
    LayeredSceneDescription,
    SceneIndexRecord,
    ScenarioClassification,
)
from utils.logging_utils import get_logger

logger = get_logger(__name__)

# Hard cap regardless of how many questions the model proposes.
_MAX_QUESTIONS = 3


def generate_questions_with_gemini(
    sample_record: SceneIndexRecord,
    description: LayeredSceneDescription,
    scenario: ScenarioClassification,
    preview_paths: dict,
    gemini_client,
    config,
) -> List[CandidateQuestion]:
    """Generate one to three deduplicated candidate questions."""
    image_paths = select_image_paths(
        preview_paths, config.llm.max_images_per_request
    )
    prompt = question_generation_prompt.build_prompt(
        sample_record, description, scenario
    )
    context = json.dumps(
        {
            "description": description.model_dump(),
            "scenario": scenario.model_dump(),
        },
        sort_keys=True,
    )
    cache_key = build_cache_key(
        sample_id=sample_record.sample_id,
        prompt_name=question_generation_prompt.PROMPT_NAME,
        prompt_version=question_generation_prompt.PROMPT_VERSION,
        model_name=gemini_client.model_name,
        image_paths=image_paths,
        context=context,
    )
    data = gemini_client.generate_json(
        prompt=prompt,
        schema=question_generation_prompt.OUTPUT_SCHEMA,
        image_paths=image_paths,
        cache_key=cache_key,
    )

    max_questions = min(
        _MAX_QUESTIONS, getattr(config.pipeline, "max_questions_per_sample", _MAX_QUESTIONS)
    )
    questions: List[CandidateQuestion] = []
    seen: set[str] = set()
    for raw in data.get("questions", []):
        question = CandidateQuestion.model_validate(raw)
        key = question.question.strip().lower()
        if key in seen:
            continue
        seen.add(key)
        questions.append(question)
        if len(questions) >= max_questions:
            break

    if not questions:
        logger.warning(
            "No questions generated for sample %s.", sample_record.sample_id
        )
    return questions
