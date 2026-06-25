"""Answerability classification with Gemini (Phase 12)."""

from __future__ import annotations

import json

from prompts import answerability_prompt
from prompts.base import build_cache_key, select_image_paths
from schema import (
    AnswerabilityLabel,
    CandidateQuestion,
    LayeredSceneDescription,
    SceneIndexRecord,
    ScenarioClassification,
)


def classify_answerability_with_gemini(
    sample_record: SceneIndexRecord,
    description: LayeredSceneDescription,
    scenario: ScenarioClassification,
    question: CandidateQuestion,
    preview_paths: dict,
    gemini_client,
    config,
) -> AnswerabilityLabel:
    """Propose the ground-truth answerability label for one question."""
    image_paths = select_image_paths(
        preview_paths, config.gemini.max_images_per_request
    )
    prompt = answerability_prompt.build_prompt(
        sample_record, description, scenario, question
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
        prompt_name=answerability_prompt.PROMPT_NAME,
        prompt_version=answerability_prompt.PROMPT_VERSION,
        model_name=gemini_client.model_name,
        image_paths=image_paths,
        context=context,
        question=question.question,
    )
    data = gemini_client.generate_json(
        prompt=prompt,
        schema=answerability_prompt.OUTPUT_SCHEMA,
        image_paths=image_paths,
        cache_key=cache_key,
    )
    return AnswerabilityLabel.model_validate(data)
