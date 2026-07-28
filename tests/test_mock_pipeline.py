"""End-to-end test of the mock annotation pipeline.

Kept separate from the README-named test files so those stay free of the
Gemini mock client. The mock makes no API calls and never reads the image
files, so fake preview paths are fine.
"""

from config import AppConfig, DatasetConfig
from schema import LayeredSceneDescription, SceneIndexRecord
from candidate_ranker import rank_candidates


def _mock_config() -> AppConfig:
    return AppConfig(dataset=DatasetConfig(dataroot="C:/none", version="v1.0-mini"))


def _mock_client(config: AppConfig):
    from llm.mock_gemini_client import MockGeminiClient
    from llm.rate_limiter import RateLimiter

    return MockGeminiClient(
        model_name="gemini-2.5-flash",
        cache=None,
        rate_limiter=RateLimiter(0.0),
        config=config.llm,
    )


def test_mock_pipeline_end_to_end_produces_valid_candidates():
    from annotator.answerability_classifier import classify_answerability_with_gemini
    from annotator.candidate_builder import build_candidate_records
    from annotator.layered_descriptor import describe_scene_with_gemini
    from annotator.question_generator import generate_questions_with_gemini
    from annotator.scenario_classifier import classify_scenario_with_gemini

    config = _mock_config()
    client = _mock_client(config)

    all_candidates = []
    for i in range(20):
        record = SceneIndexRecord(
            sample_id=f"sample_{i}",
            scene_token=f"scene_{i}",
            timestamp=i,
            camera_paths={"CAM_FRONT": f"img_{i}.jpg"},
            num_annotations=0,
        )
        # Fake preview paths (the mock never opens them).
        preview_paths = {
            "CAM_FRONT": f"previews/CAM_FRONT_{i:04d}.jpg",
            "CAM_FRONT_LEFT": f"previews/CAM_FRONT_LEFT_{i:04d}.jpg",
        }
        description = describe_scene_with_gemini(record, preview_paths, client, config)
        assert isinstance(description, LayeredSceneDescription)

        scenario = classify_scenario_with_gemini(
            record, description, preview_paths, client, config
        )
        if not scenario.is_relevant or "not_relevant" in scenario.scenario_clusters:
            continue

        questions = generate_questions_with_gemini(
            record, description, scenario, preview_paths, client, config
        )
        assert 1 <= len(questions) <= 3

        labels = [
            classify_answerability_with_gemini(
                record, description, scenario, q, preview_paths, client, config
            )
            for q in questions
        ]
        candidates = build_candidate_records(
            record,
            preview_paths,
            description,
            scenario,
            questions,
            labels,
            client.model_name,
        )
        all_candidates.extend(candidates)

    # The mock yields a mix of relevant scenarios, so we get candidates.
    assert all_candidates
    ranked = rank_candidates(all_candidates)
    assert ranked == sorted(ranked, key=lambda c: c.priority_score, reverse=True)
    # Mock calls do not count against the paid-call budget.
    assert client.call_count > 0
