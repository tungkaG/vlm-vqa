"""Answerability-classification prompt (Phase 8 / Phase 12).

Asks Gemini to propose the ground-truth answerability label for a single
candidate question. Output validates as
:class:`schema.AnswerabilityLabel`, including the abstention-consistency
rule (abstention_required is true exactly when the label is unanswerable
or ambiguous).
"""

from __future__ import annotations

import json

from constants import ANSWERABILITY_LABELS, SAFETY_ACTIONS, UNCERTAINTY_SOURCES
from schema import (
    CandidateQuestion,
    LayeredSceneDescription,
    SceneIndexRecord,
    ScenarioClassification,
)

PROMPT_NAME = "answerability_classification"
PROMPT_VERSION = "v1"

OUTPUT_SCHEMA = {
    "title": "AnswerabilityLabel",
    "type": "object",
    "required": [
        "answerability",
        "ground_truth_answer",
        "abstention_required",
        "visible_evidence",
        "missing_evidence",
        "uncertainty_source",
        "recommended_action",
        "rationale",
    ],
    "properties": {
        "answerability": {"type": "string", "enum": ANSWERABILITY_LABELS},
        "ground_truth_answer": {"type": "string"},
        "abstention_required": {"type": "boolean"},
        "visible_evidence": {"type": "string"},
        "missing_evidence": {"type": "string"},
        "uncertainty_source": {"type": "string", "enum": UNCERTAINTY_SOURCES},
        "recommended_action": {"type": "string", "enum": SAFETY_ACTIONS},
        "rationale": {"type": "string"},
    },
}

FEW_SHOT_EXAMPLES = [
    {
        "answerability": "unanswerable",
        "ground_truth_answer": "Cannot determine from the available visual evidence.",
        "abstention_required": True,
        "visible_evidence": "A large vehicle is visible near the intersection.",
        "missing_evidence": "The traffic light state is not visible.",
        "uncertainty_source": "occlusion",
        "recommended_action": "slow_down",
        "rationale": "The required traffic light color is not visible, so the "
        "model should not guess.",
    },
    {
        "answerability": "answerable",
        "ground_truth_answer": "Yes, one pedestrian is waiting at the crosswalk.",
        "abstention_required": False,
        "visible_evidence": "A pedestrian is clearly standing at the crosswalk.",
        "missing_evidence": "",
        "uncertainty_source": "not_uncertain",
        "recommended_action": "proceed",
        "rationale": "The pedestrian is fully visible, so the question is answerable.",
    },
]

_INSTRUCTIONS = (
    "You are assigning the ground-truth answerability label for one visual "
    "question about an autonomous-driving scene. Use only the images, the "
    "layered description and the scenario classification below.\n\n"
    "Decide:\n"
    "- answerability: 'answerable' if the question can be answered "
    "confidently from the images; 'unanswerable' if required evidence is "
    "missing/occluded/illegible; 'ambiguous' if the evidence is conflicting "
    "or the answer depends on unknowable future intent.\n"
    "- ground_truth_answer: a concise correct answer, OR an explicit "
    "abstention statement when the label is unanswerable/ambiguous. It must "
    "never be empty.\n"
    "- abstention_required: MUST be true when answerability is 'unanswerable' "
    "or 'ambiguous', and false when 'answerable'.\n"
    "- uncertainty_source: the dominant reason (use 'not_uncertain' for "
    "answerable questions).\n"
    "- recommended_action: the safe driving action implied.\n\n"
    f"Allowed answerability: {ANSWERABILITY_LABELS}\n"
    f"Allowed uncertainty_source: {UNCERTAINTY_SOURCES}\n"
    f"Allowed recommended_action: {SAFETY_ACTIONS}\n\n"
    "Return ONLY a single JSON object with keys answerability, "
    "ground_truth_answer, abstention_required, visible_evidence, "
    "missing_evidence, uncertainty_source, recommended_action, rationale. "
    "No prose, no Markdown fences.\n"
)


def build_prompt(
    sample_record: SceneIndexRecord,
    description: LayeredSceneDescription,
    scenario: ScenarioClassification,
    question: CandidateQuestion,
) -> str:
    """Build the answerability-classification prompt for one question."""
    description_text = json.dumps(description.model_dump(), indent=2)
    scenario_text = json.dumps(scenario.model_dump(), indent=2)
    question_text = json.dumps(question.model_dump(), indent=2)
    example_text = json.dumps(FEW_SHOT_EXAMPLES, indent=2)
    return (
        f"{_INSTRUCTIONS}\n"
        f"Layered scene description:\n{description_text}\n\n"
        f"Scenario classification:\n{scenario_text}\n\n"
        f"Question to label:\n{question_text}\n\n"
        f"Examples of well-formed responses:\n{example_text}\n"
    )
