"""Layered scene description with Gemini (Phase 9)."""

from __future__ import annotations

from prompts import layered_scene_prompt
from prompts.base import build_cache_key, select_image_paths
from schema import LayeredSceneDescription, SceneIndexRecord


def describe_scene_with_gemini(
    sample_record: SceneIndexRecord,
    preview_paths: dict,
    gemini_client,
    config,
) -> LayeredSceneDescription:
    """Ask Gemini for a layered semantic description of one sample."""
    image_paths = select_image_paths(
        preview_paths, config.gemini.max_images_per_request
    )
    prompt = layered_scene_prompt.build_prompt(sample_record)
    cache_key = build_cache_key(
        sample_id=sample_record.sample_id,
        prompt_name=layered_scene_prompt.PROMPT_NAME,
        prompt_version=layered_scene_prompt.PROMPT_VERSION,
        model_name=gemini_client.model_name,
        image_paths=image_paths,
    )
    data = gemini_client.generate_json(
        prompt=prompt,
        schema=layered_scene_prompt.OUTPUT_SCHEMA,
        image_paths=image_paths,
        cache_key=cache_key,
    )
    return LayeredSceneDescription.model_validate(data)
