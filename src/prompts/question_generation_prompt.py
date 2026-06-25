"""Question-generation prompt (Phase 8 / Phase 11).

Asks Gemini to generate one to three VQA questions that test whether a
vision-language model should answer or abstain. Each question validates
as :class:`schema.CandidateQuestion`.
"""

from __future__ import annotations

import json

from constants import ANSWERABILITY_LABELS, TASK_LAYERS
from schema import LayeredSceneDescription, SceneIndexRecord, ScenarioClassification

PROMPT_NAME = "question_generation"
PROMPT_VERSION = "v1"

OUTPUT_SCHEMA = {
    "title": "QuestionList",
    "type": "object",
    "required": ["questions"],
    "properties": {
        "questions": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["question", "task_layer"],
                "properties": {
                    "question": {"type": "string"},
                    "target_object": {"type": "string"},
                    "task_layer": {"type": "string", "enum": TASK_LAYERS},
                    "expected_answerability": {
                        "type": "string",
                        "enum": ANSWERABILITY_LABELS,
                    },
                },
            },
        }
    },
}

FEW_SHOT_EXAMPLES = [
    {
        "questions": [
            {
                "question": "What color is the traffic light controlling the ego lane?",
                "target_object": "traffic light",
                "task_layer": "infrastructure",
                "expected_answerability": "unanswerable",
            },
            {
                "question": "Is there a pedestrian waiting at the crosswalk ahead?",
                "target_object": "pedestrian",
                "task_layer": "perception",
                "expected_answerability": "answerable",
            },
        ]
    }
]

_INSTRUCTIONS = (
    "You are generating visual question-answering items for autonomous "
    "driving. The purpose is to test whether a vision-language model should "
    "ANSWER or ABSTAIN.\n\n"
    "Using the images, the layered scene description and the scenario "
    "classification below, generate between one and three high-quality "
    "questions. Rules:\n"
    "- Each question must be specific and grounded in this scene.\n"
    "- Link each question to the most relevant task_layer.\n"
    "- Do NOT generate vague questions.\n"
    "- Do NOT require information outside the images UNLESS the intended "
    "label is 'unanswerable' or 'ambiguous'.\n"
    "- For relevant scenes, include at least one question whose correct "
    "answer is to abstain (unanswerable or ambiguous) when the scenario "
    "involves occlusion, sensor degradation, ambiguous intent or missing "
    "views.\n\n"
    f"Allowed task_layer: {TASK_LAYERS}\n"
    f"Allowed expected_answerability: {ANSWERABILITY_LABELS}\n\n"
    "Return ONLY a single JSON object of the form "
    '{"questions": [{"question": ..., "target_object": ..., '
    '"task_layer": ..., "expected_answerability": ...}, ...]}. '
    "No prose, no Markdown fences.\n"
)


def build_prompt(
    sample_record: SceneIndexRecord,
    description: LayeredSceneDescription,
    scenario: ScenarioClassification,
) -> str:
    """Build the question-generation prompt for one sample."""
    description_text = json.dumps(description.model_dump(), indent=2)
    scenario_text = json.dumps(scenario.model_dump(), indent=2)
    example_text = json.dumps(FEW_SHOT_EXAMPLES[0], indent=2)
    return (
        f"{_INSTRUCTIONS}\n"
        f"Layered scene description:\n{description_text}\n\n"
        f"Scenario classification:\n{scenario_text}\n\n"
        f"Example of a well-formed response:\n{example_text}\n"
    )
