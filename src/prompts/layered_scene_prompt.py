"""Layered scene-description prompt (Phase 8 / Phase 9).

Asks Gemini to describe a nuScenes sample using five structured semantic
layers (street, infrastructure, movable objects, environment,
uncertainty). Output is strict JSON validated as
:class:`schema.LayeredSceneDescription`.
"""

from __future__ import annotations

from schema import SceneIndexRecord

PROMPT_NAME = "layered_scene_description"
PROMPT_VERSION = "v1"

# Minimal JSON-schema subset (see utils.json_utils.validate_json_schema).
# The "title" drives mock dispatch; it is ignored by the validator.
OUTPUT_SCHEMA = {
    "title": "LayeredSceneDescription",
    "type": "object",
    "required": [
        "street",
        "infrastructure",
        "movable_objects",
        "environment",
        "uncertainty",
    ],
    "properties": {
        "street": {
            "type": "object",
            "required": [
                "road_layout",
                "lanes_visible",
                "crosswalk_visible",
                "occluded_regions",
            ],
            "properties": {
                "road_layout": {"type": "string"},
                "lanes_visible": {"type": "string"},
                "crosswalk_visible": {"type": "string"},
                "occluded_regions": {"type": "array", "items": {"type": "string"}},
            },
        },
        "infrastructure": {
            "type": "object",
            "required": [
                "traffic_lights",
                "traffic_signs",
                "barriers_or_construction",
                "visibility_issues",
            ],
            "properties": {
                "traffic_lights": {"type": "array", "items": {"type": "string"}},
                "traffic_signs": {"type": "array", "items": {"type": "string"}},
                "barriers_or_construction": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "visibility_issues": {"type": "array", "items": {"type": "string"}},
            },
        },
        "movable_objects": {
            "type": "object",
            "required": [
                "vehicles",
                "pedestrians",
                "cyclists",
                "ambiguous_intentions",
            ],
            "properties": {
                "vehicles": {"type": "array", "items": {"type": "string"}},
                "pedestrians": {"type": "array", "items": {"type": "string"}},
                "cyclists": {"type": "array", "items": {"type": "string"}},
                "ambiguous_intentions": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
        },
        "environment": {
            "type": "object",
            "required": ["weather", "lighting", "visibility", "image_quality_issues"],
            "properties": {
                "weather": {"type": "string"},
                "lighting": {"type": "string"},
                "visibility": {"type": "string"},
                "image_quality_issues": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
        },
        "uncertainty": {
            "type": "object",
            "required": [
                "possible_occlusion",
                "possible_sensor_degradation",
                "possible_ambiguous_intent",
                "multi_view_needed",
                "uncertainty_reason",
            ],
            "properties": {
                "possible_occlusion": {"type": "boolean"},
                "possible_sensor_degradation": {"type": "boolean"},
                "possible_ambiguous_intent": {"type": "boolean"},
                "multi_view_needed": {"type": "boolean"},
                "uncertainty_reason": {"type": "string"},
            },
        },
    },
}

FEW_SHOT_EXAMPLES = [
    {
        "street": {
            "road_layout": "Four-way urban intersection with the ego vehicle "
            "stopped in the leftmost through lane.",
            "lanes_visible": "Three forward lanes and one left-turn lane visible.",
            "crosswalk_visible": "Marked crosswalk visible directly ahead.",
            "occluded_regions": ["Area behind the parked truck on the right"],
        },
        "infrastructure": {
            "traffic_lights": ["Traffic light ahead, state not clearly visible"],
            "traffic_signs": ["No-parking sign on the right"],
            "barriers_or_construction": ["Temporary construction barriers on right"],
            "visibility_issues": ["Traffic light partially occluded by a truck"],
        },
        "movable_objects": {
            "vehicles": ["Parked truck right", "White van crossing ahead"],
            "pedestrians": ["Pedestrian waiting at the crosswalk"],
            "cyclists": [],
            "ambiguous_intentions": ["Pedestrian may be about to cross"],
        },
        "environment": {
            "weather": "Clear",
            "lighting": "Daylight",
            "visibility": "Good overall, reduced near the occluded right side",
            "image_quality_issues": [],
        },
        "uncertainty": {
            "possible_occlusion": True,
            "possible_sensor_degradation": False,
            "possible_ambiguous_intent": True,
            "multi_view_needed": False,
            "uncertainty_reason": "The traffic light state is obscured by a truck.",
        },
    }
]

_INSTRUCTIONS = (
    "You are an autonomous-driving perception assistant. You receive up to "
    "six surround-view camera images (front-left, front, front-right, "
    "back-left, back, back-right) captured at the same instant from a "
    "self-driving car.\n\n"
    "Describe the scene strictly from what is visible in the images using "
    "five semantic layers: street, infrastructure, movable_objects, "
    "environment and uncertainty. Do not invent objects that are not "
    "visible. When information is missing, occluded, blurred or would "
    "require a viewpoint you do not have, record that in the uncertainty "
    "layer and in the relevant occlusion/visibility fields.\n\n"
    "Return ONLY a single JSON object matching this exact shape (no prose, "
    "no Markdown fences):\n"
)


def build_prompt(sample_record: SceneIndexRecord) -> str:
    """Build the layered scene-description prompt for one sample."""
    import json

    hints = ""
    if sample_record.scene_description:
        hints += f"\nScene context (for orientation only): {sample_record.scene_description}"
    if sample_record.annotation_categories:
        top = ", ".join(sample_record.annotation_categories[:12])
        hints += f"\nObject categories present in the dataset annotations: {top}"
    hints += (
        "\nTreat the context above only as a hint; base every field strictly "
        "on what is actually visible in the images."
    )

    schema_text = json.dumps(OUTPUT_SCHEMA["properties"], indent=2)
    example_text = json.dumps(FEW_SHOT_EXAMPLES[0], indent=2)
    return (
        f"{_INSTRUCTIONS}{schema_text}\n\n"
        f"Example of a well-formed response:\n{example_text}\n"
        f"{hints}"
    )
