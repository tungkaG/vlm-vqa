"""Project-wide constants and controlled vocabularies.

These lists are the single source of truth for the enums that the
schema models and annotator stages validate against.
"""

from __future__ import annotations

# The six nuScenes surround-view cameras.
CAMERA_NAMES = [
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
]

# Thesis scenario clusters (README "Main Scenario Clusters").
SCENARIO_CLUSTERS = [
    "normal_answerable_control",
    "object_occlusion",
    "traffic_light_or_sign_occlusion",
    "sensor_degradation",
    "ambiguous_agent_intent",
    "planning_under_occlusion",
    "risk_under_incomplete_evidence",
    "multi_view_required",
    "not_relevant",
]

# Answerability labels for generated questions.
ANSWERABILITY_LABELS = [
    "answerable",
    "unanswerable",
    "ambiguous",
]

# Allowed uncertainty sources.
UNCERTAINTY_SOURCES = [
    "occlusion",
    "sensor_degradation",
    "multi_view_missing",
    "future_intent_unknown",
    "insufficient_resolution",
    "conflicting_visual_evidence",
    "not_uncertain",
]

# Allowed recommended safety actions.
SAFETY_ACTIONS = [
    "proceed",
    "slow_down",
    "stop_or_wait",
    "minimal_risk_response",
]

# Semantic task layers a question can target.
TASK_LAYERS = [
    "perception",
    "infrastructure",
    "prediction",
    "planning",
    "environment",
]
