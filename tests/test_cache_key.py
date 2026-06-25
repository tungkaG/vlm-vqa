"""Tests for cache-key construction (Phase 19).

Pure local logic; no Gemini calls and no mock client.
"""

from prompts.base import build_cache_key


def _key(**overrides) -> str:
    params = dict(
        sample_id="sample_1",
        prompt_name="layered_scene_description",
        prompt_version="v1",
        model_name="gemini-2.5-flash",
        image_paths=["previews/CAM_FRONT_aaaa.jpg", "previews/CAM_BACK_bbbb.jpg"],
    )
    params.update(overrides)
    return build_cache_key(**params)


def test_same_input_produces_same_key():
    assert _key() == _key()


def test_image_order_does_not_change_key():
    a = _key(image_paths=["previews/CAM_FRONT_aaaa.jpg", "previews/CAM_BACK_bbbb.jpg"])
    b = _key(image_paths=["previews/CAM_BACK_bbbb.jpg", "previews/CAM_FRONT_aaaa.jpg"])
    assert a == b


def test_different_prompt_version_produces_different_key():
    assert _key(prompt_version="v1") != _key(prompt_version="v2")


def test_different_prompt_name_produces_different_key():
    assert _key(prompt_name="layered_scene_description") != _key(
        prompt_name="scenario_classification"
    )


def test_different_model_produces_different_key():
    assert _key(model_name="gemini-2.5-flash") != _key(model_name="gemini-2.5-pro")


def test_different_images_produce_different_key():
    assert _key() != _key(image_paths=["previews/CAM_FRONT_cccc.jpg"])


def test_question_changes_key():
    assert _key() != _key(question="What color is the light?")


def test_context_changes_key():
    assert _key() != _key(context="some-upstream-context")
