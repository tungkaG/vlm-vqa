"""Scenario classification with Gemini (Phase 10)."""

from __future__ import annotations

import json

from prompts import scenario_prompt
from prompts.base import build_cache_key, select_image_paths
from schema import LayeredSceneDescription, SceneIndexRecord, ScenarioClassification


def classify_scenario_with_gemini(
    sample_record: SceneIndexRecord,
    description: LayeredSceneDescription,
    preview_paths: dict,
    gemini_client,
    config,
) -> ScenarioClassification:
    """Classify a sample's relevance and scenario clusters."""
    image_paths = select_image_paths(
        preview_paths, config.gemini.max_images_per_request
    )
    prompt = scenario_prompt.build_prompt(sample_record, description)
    cache_key = build_cache_key(
        sample_id=sample_record.sample_id,
        prompt_name=scenario_prompt.PROMPT_NAME,
        prompt_version=scenario_prompt.PROMPT_VERSION,
        model_name=gemini_client.model_name,
        image_paths=image_paths,
        context=json.dumps(description.model_dump(), sort_keys=True),
    )
    data = gemini_client.generate_json(
        prompt=prompt,
        schema=scenario_prompt.OUTPUT_SCHEMA,
        image_paths=image_paths,
        cache_key=cache_key,
    )
    return ScenarioClassification.model_validate(data)
