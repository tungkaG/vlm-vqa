"""Tests for the Pydantic data models (Phase 4 acceptance criteria).

These are pure local validation tests; no external API is called.
"""

import pytest
from pydantic import ValidationError

from schema import (
    AnswerabilityLabel,
    CandidateRecord,
    LayeredSceneDescription,
    ScenarioClassification,
)


def _valid_answerable_label() -> AnswerabilityLabel:
    return AnswerabilityLabel(
        answerability="answerable",
        ground_truth_answer="Yes, a vehicle is ahead.",
        abstention_required=False,
        visible_evidence="A car is clearly visible in the ego lane.",
        missing_evidence="",
        uncertainty_source="not_uncertain",
        recommended_action="proceed",
        rationale="The vehicle is fully visible.",
    )


def _valid_unanswerable_label() -> AnswerabilityLabel:
    return AnswerabilityLabel(
        answerability="unanswerable",
        ground_truth_answer="Cannot determine from the available visual evidence.",
        abstention_required=True,
        visible_evidence="A large vehicle is near the intersection.",
        missing_evidence="The traffic light state is not visible.",
        uncertainty_source="occlusion",
        recommended_action="slow_down",
        rationale="The traffic light is occluded.",
    )


def _layered_description() -> LayeredSceneDescription:
    return LayeredSceneDescription.model_validate(
        {
            "street": {},
            "infrastructure": {},
            "movable_objects": {},
            "environment": {},
            "uncertainty": {},
        }
    )


def _scenario() -> ScenarioClassification:
    return ScenarioClassification(
        is_relevant=True,
        scenario_clusters=["object_occlusion"],
        task_layer="perception",
        uncertainty_sources=["occlusion"],
        safety_relevance="May require abstention.",
        reason="Object partly hidden.",
    )


def test_answerability_label_round_trip():
    label = _valid_unanswerable_label()
    payload = label.model_dump_json()
    restored = AnswerabilityLabel.model_validate_json(payload)
    assert restored == label


def test_invalid_answerability_raises():
    with pytest.raises(ValidationError):
        AnswerabilityLabel(
            answerability="definitely",  # invalid
            ground_truth_answer="x",
            abstention_required=True,
            uncertainty_source="occlusion",
            recommended_action="slow_down",
        )


def test_invalid_safety_action_raises():
    with pytest.raises(ValidationError):
        AnswerabilityLabel(
            answerability="unanswerable",
            ground_truth_answer="x",
            abstention_required=True,
            uncertainty_source="occlusion",
            recommended_action="floor_it",  # invalid
        )


def test_invalid_uncertainty_source_raises():
    with pytest.raises(ValidationError):
        AnswerabilityLabel(
            answerability="unanswerable",
            ground_truth_answer="x",
            abstention_required=True,
            uncertainty_source="vibes",  # invalid
            recommended_action="slow_down",
        )


def test_empty_ground_truth_answer_raises():
    with pytest.raises(ValidationError):
        AnswerabilityLabel(
            answerability="answerable",
            ground_truth_answer="   ",  # blank
            abstention_required=False,
            uncertainty_source="not_uncertain",
            recommended_action="proceed",
        )


def test_abstention_must_match_answerability():
    # answerable but abstention_required=True is inconsistent.
    with pytest.raises(ValidationError):
        AnswerabilityLabel(
            answerability="answerable",
            ground_truth_answer="Yes.",
            abstention_required=True,
            uncertainty_source="not_uncertain",
            recommended_action="proceed",
        )
    # unanswerable but abstention_required=False is inconsistent.
    with pytest.raises(ValidationError):
        AnswerabilityLabel(
            answerability="unanswerable",
            ground_truth_answer="Cannot determine.",
            abstention_required=False,
            uncertainty_source="occlusion",
            recommended_action="slow_down",
        )


def test_invalid_scenario_cluster_raises():
    with pytest.raises(ValidationError):
        ScenarioClassification(
            is_relevant=True,
            scenario_clusters=["not_a_real_cluster"],
            task_layer="perception",
        )


def test_empty_scenario_clusters_raises():
    with pytest.raises(ValidationError):
        ScenarioClassification(
            is_relevant=True,
            scenario_clusters=[],
            task_layer="perception",
        )


def test_candidate_record_round_trip():
    record = CandidateRecord(
        sample_id="sample_token",
        scene_token="scene_token",
        camera_paths={"CAM_FRONT": "front.jpg", "CAM_BACK": None},
        preview_paths={"CAM_FRONT": "front_preview.jpg"},
        layered_scene_description=_layered_description(),
        scenario_classification=_scenario(),
        question="What color is the traffic light?",
        answerability_label=_valid_unanswerable_label(),
        priority_score=4.0,
        source_model_name="gemini-2.5-flash",
        human_verified=False,
        human_notes=None,
    )
    payload = record.model_dump_json()
    restored = CandidateRecord.model_validate_json(payload)
    assert restored == record
    assert restored.answerability_label.answerability == "unanswerable"


def test_answerable_label_is_valid():
    label = _valid_answerable_label()
    assert label.abstention_required is False
