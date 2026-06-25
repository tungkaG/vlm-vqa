"""Candidate record builder (Phase 13)."""

from __future__ import annotations

from typing import List

from schema import (
    AnswerabilityLabel,
    CandidateQuestion,
    CandidateRecord,
    LayeredSceneDescription,
    SceneIndexRecord,
    ScenarioClassification,
)


def build_candidate_records(
    sample_record: SceneIndexRecord,
    preview_paths: dict,
    description: LayeredSceneDescription,
    scenario: ScenarioClassification,
    questions: List[CandidateQuestion],
    answerability_labels: List[AnswerabilityLabel],
    model_name: str,
) -> List[CandidateRecord]:
    """Combine per-sample outputs into one validated record per question."""
    if len(questions) != len(answerability_labels):
        raise ValueError(
            f"questions ({len(questions)}) and answerability_labels "
            f"({len(answerability_labels)}) must have equal length."
        )

    records: List[CandidateRecord] = []
    for question, label in zip(questions, answerability_labels):
        records.append(
            CandidateRecord(
                sample_id=sample_record.sample_id,
                scene_token=sample_record.scene_token,
                camera_paths=sample_record.camera_paths,
                preview_paths=preview_paths,
                layered_scene_description=description,
                scenario_classification=scenario,
                question=question.question,
                answerability_label=label,
                priority_score=0.0,
                source_model_name=model_name,
                human_verified=False,
            )
        )
    return records
