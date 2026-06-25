"""Scenario-classification prompt (Phase 8 / Phase 10).

Asks Gemini to decide whether a scene is relevant to the thesis and to
assign one or more scenario clusters, a task layer and uncertainty
sources. Output validates as :class:`schema.ScenarioClassification`.
"""

from __future__ import annotations

import json

from constants import SCENARIO_CLUSTERS, TASK_LAYERS, UNCERTAINTY_SOURCES
from schema import LayeredSceneDescription, SceneIndexRecord

PROMPT_NAME = "scenario_classification"
PROMPT_VERSION = "v1"

OUTPUT_SCHEMA = {
    "title": "ScenarioClassification",
    "type": "object",
    "required": [
        "is_relevant",
        "scenario_clusters",
        "task_layer",
        "uncertainty_sources",
        "safety_relevance",
        "reason",
    ],
    "properties": {
        "is_relevant": {"type": "boolean"},
        "scenario_clusters": {
            "type": "array",
            "items": {"type": "string", "enum": SCENARIO_CLUSTERS},
        },
        "task_layer": {"type": "string", "enum": TASK_LAYERS},
        "uncertainty_sources": {
            "type": "array",
            "items": {"type": "string", "enum": UNCERTAINTY_SOURCES},
        },
        "safety_relevance": {"type": "string"},
        "reason": {"type": "string"},
    },
}

FEW_SHOT_EXAMPLES = [
    {
        "is_relevant": True,
        "scenario_clusters": ["traffic_light_or_sign_occlusion"],
        "task_layer": "infrastructure",
        "uncertainty_sources": ["occlusion"],
        "safety_relevance": "The model may need to abstain because the traffic "
        "light state is hidden by a truck.",
        "reason": "A traffic light controlling the ego lane is partially occluded.",
    },
    {
        "is_relevant": False,
        "scenario_clusters": ["not_relevant"],
        "task_layer": "perception",
        "uncertainty_sources": ["not_uncertain"],
        "safety_relevance": "Ordinary open road with no uncertainty of interest.",
        "reason": "Clear empty road with full visibility and no edge case.",
    },
]

_INSTRUCTIONS = (
    "You are labelling autonomous-driving scenes for a thesis about when a "
    "vision-language model should ANSWER versus ABSTAIN.\n\n"
    "Given the surround-view camera images and the layered scene "
    "description below, decide:\n"
    "1. is_relevant: true if the scene contains uncertainty, occlusion, "
    "sensor issues, ambiguous intent or planning-under-incomplete-evidence "
    "that is interesting for abstention; false for ordinary fully-observable "
    "scenes.\n"
    "2. scenario_clusters: one or more clusters from the allowed list. Use "
    "exactly 'not_relevant' (and only that) when is_relevant is false.\n"
    "3. task_layer: the single most relevant task layer.\n"
    "4. uncertainty_sources: zero or more from the allowed list (use "
    "'not_uncertain' when there is none).\n\n"
    f"Allowed scenario_clusters: {SCENARIO_CLUSTERS}\n"
    f"Allowed task_layer: {TASK_LAYERS}\n"
    f"Allowed uncertainty_sources: {UNCERTAINTY_SOURCES}\n\n"
    "Return ONLY a single JSON object with keys is_relevant, "
    "scenario_clusters, task_layer, uncertainty_sources, safety_relevance, "
    "reason. No prose, no Markdown fences.\n"
)


def build_prompt(
    sample_record: SceneIndexRecord, description: LayeredSceneDescription
) -> str:
    """Build the scenario-classification prompt for one sample."""
    description_text = json.dumps(description.model_dump(), indent=2)
    example_text = json.dumps(FEW_SHOT_EXAMPLES, indent=2)
    return (
        f"{_INSTRUCTIONS}\n"
        f"Layered scene description:\n{description_text}\n\n"
        f"Examples of well-formed responses:\n{example_text}\n"
    )
