"""Strict Pydantic data models for every pipeline record (Phase 4).

These models validate all intermediate and final records. Controlled
vocabularies are enforced against the lists in :mod:`constants`, and the
answerability/abstention consistency rule is enforced by a model
validator so invalid labels raise clear errors.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

from constants import (
    ANSWERABILITY_LABELS,
    SAFETY_ACTIONS,
    SCENARIO_CLUSTERS,
    TASK_LAYERS,
    UNCERTAINTY_SOURCES,
)

# Answerability values that require the model to abstain.
_ABSTAIN_LABELS = {"unanswerable", "ambiguous"}


# --------------------------------------------------------------------------
# Scene index + Gemini call bookkeeping
# --------------------------------------------------------------------------
class SceneIndexRecord(BaseModel):
    sample_id: str
    scene_token: str
    timestamp: int
    camera_paths: Dict[str, Optional[str]]
    num_annotations: int
    annotation_categories: List[str] = Field(default_factory=list)
    category_counts: Dict[str, int] = Field(default_factory=dict)
    scene_description: Optional[str] = None


class GeminiCallRecord(BaseModel):
    sample_id: str
    prompt_name: str
    prompt_version: str
    model_name: str
    cache_key: str
    response_json: dict
    raw_text: Optional[str] = None


# --------------------------------------------------------------------------
# Layered scene description (Phase 9 output shape)
# --------------------------------------------------------------------------
class StreetLayer(BaseModel):
    road_layout: str = ""
    lanes_visible: str = ""
    crosswalk_visible: str = ""
    occluded_regions: List[str] = Field(default_factory=list)


class InfrastructureLayer(BaseModel):
    traffic_lights: List[str] = Field(default_factory=list)
    traffic_signs: List[str] = Field(default_factory=list)
    barriers_or_construction: List[str] = Field(default_factory=list)
    visibility_issues: List[str] = Field(default_factory=list)


class MovableObjectsLayer(BaseModel):
    vehicles: List[str] = Field(default_factory=list)
    pedestrians: List[str] = Field(default_factory=list)
    cyclists: List[str] = Field(default_factory=list)
    ambiguous_intentions: List[str] = Field(default_factory=list)


class EnvironmentLayer(BaseModel):
    weather: str = ""
    lighting: str = ""
    visibility: str = ""
    image_quality_issues: List[str] = Field(default_factory=list)


class UncertaintyLayer(BaseModel):
    possible_occlusion: bool = False
    possible_sensor_degradation: bool = False
    possible_ambiguous_intent: bool = False
    multi_view_needed: bool = False
    uncertainty_reason: str = ""


class LayeredSceneDescription(BaseModel):
    street: StreetLayer
    infrastructure: InfrastructureLayer
    movable_objects: MovableObjectsLayer
    environment: EnvironmentLayer
    uncertainty: UncertaintyLayer


# --------------------------------------------------------------------------
# Scenario classification
# --------------------------------------------------------------------------
class ScenarioClassification(BaseModel):
    is_relevant: bool
    scenario_clusters: List[str]
    task_layer: str
    uncertainty_sources: List[str] = Field(default_factory=list)
    safety_relevance: str = ""
    reason: str = ""

    @field_validator("scenario_clusters")
    @classmethod
    def _validate_clusters(cls, value: List[str]) -> List[str]:
        if not value:
            raise ValueError("scenario_clusters must contain at least one cluster.")
        invalid = [c for c in value if c not in SCENARIO_CLUSTERS]
        if invalid:
            raise ValueError(
                f"Invalid scenario cluster(s) {invalid}. "
                f"Allowed: {SCENARIO_CLUSTERS}"
            )
        return value

    @field_validator("task_layer")
    @classmethod
    def _validate_task_layer(cls, value: str) -> str:
        if value not in TASK_LAYERS:
            raise ValueError(f"Invalid task_layer '{value}'. Allowed: {TASK_LAYERS}")
        return value

    @field_validator("uncertainty_sources")
    @classmethod
    def _validate_uncertainty_sources(cls, value: List[str]) -> List[str]:
        invalid = [s for s in value if s not in UNCERTAINTY_SOURCES]
        if invalid:
            raise ValueError(
                f"Invalid uncertainty source(s) {invalid}. "
                f"Allowed: {UNCERTAINTY_SOURCES}"
            )
        return value


# --------------------------------------------------------------------------
# Candidate question
# --------------------------------------------------------------------------
class CandidateQuestion(BaseModel):
    question: str
    target_object: Optional[str] = None
    task_layer: str
    expected_answerability: Optional[str] = None

    @field_validator("question")
    @classmethod
    def _validate_question(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("question must not be empty.")
        return value

    @field_validator("task_layer")
    @classmethod
    def _validate_task_layer(cls, value: str) -> str:
        if value not in TASK_LAYERS:
            raise ValueError(f"Invalid task_layer '{value}'. Allowed: {TASK_LAYERS}")
        return value

    @field_validator("expected_answerability")
    @classmethod
    def _validate_expected(cls, value: Optional[str]) -> Optional[str]:
        if value is not None and value not in ANSWERABILITY_LABELS:
            raise ValueError(
                f"Invalid expected_answerability '{value}'. "
                f"Allowed: {ANSWERABILITY_LABELS}"
            )
        return value


# --------------------------------------------------------------------------
# Answerability label
# --------------------------------------------------------------------------
class AnswerabilityLabel(BaseModel):
    answerability: str
    ground_truth_answer: str
    abstention_required: bool
    visible_evidence: str = ""
    missing_evidence: str = ""
    uncertainty_source: str
    recommended_action: str
    rationale: str = ""

    @field_validator("answerability")
    @classmethod
    def _validate_answerability(cls, value: str) -> str:
        if value not in ANSWERABILITY_LABELS:
            raise ValueError(
                f"Invalid answerability '{value}'. Allowed: {ANSWERABILITY_LABELS}"
            )
        return value

    @field_validator("ground_truth_answer")
    @classmethod
    def _validate_answer(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("ground_truth_answer must not be empty.")
        return value

    @field_validator("uncertainty_source")
    @classmethod
    def _validate_uncertainty_source(cls, value: str) -> str:
        if value not in UNCERTAINTY_SOURCES:
            raise ValueError(
                f"Invalid uncertainty_source '{value}'. "
                f"Allowed: {UNCERTAINTY_SOURCES}"
            )
        return value

    @field_validator("recommended_action")
    @classmethod
    def _validate_action(cls, value: str) -> str:
        if value not in SAFETY_ACTIONS:
            raise ValueError(
                f"Invalid recommended_action '{value}'. Allowed: {SAFETY_ACTIONS}"
            )
        return value

    @model_validator(mode="after")
    def _validate_abstention_consistency(self) -> "AnswerabilityLabel":
        should_abstain = self.answerability in _ABSTAIN_LABELS
        if self.abstention_required != should_abstain:
            raise ValueError(
                f"abstention_required={self.abstention_required} is inconsistent "
                f"with answerability='{self.answerability}' "
                f"(expected {should_abstain})."
            )
        return self


# --------------------------------------------------------------------------
# Candidate record (combined output)
# --------------------------------------------------------------------------
class CandidateRecord(BaseModel):
    sample_id: str
    scene_token: str
    camera_paths: Dict[str, Optional[str]]
    preview_paths: Dict[str, Optional[str]]
    layered_scene_description: LayeredSceneDescription
    scenario_classification: ScenarioClassification
    question: str
    answerability_label: AnswerabilityLabel
    priority_score: float = 0.0
    source_model_name: str
    human_verified: bool = False
    human_notes: Optional[str] = None
