"""Mock Gemini client (development only).

Provides a drop-in replacement for :class:`GeminiClient` that returns
schema-conforming canned JSON instead of calling the API. It overrides
only :meth:`_produce_raw_text`; every other behaviour (caching, call
budget, rate limiting, JSON parsing, schema validation) runs through the
shared :class:`BaseGeminiClient`, so the surrounding "LLM call" is
identical to the real client.

Outputs are deterministic per sample: a seed derived from the sorted
preview file names selects a scenario archetype, so each sample yields a
coherent description, scenario, questions and answerability labels. This
gives a realistic spread for exercising candidate building and ranking.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

from llm.base_client import BaseLLMClient
from utils.hash_utils import short_hash
from utils.logging_utils import get_logger

logger = get_logger(__name__)


# Each archetype is internally consistent: the scenario clusters, the
# dominant answerability label, the uncertainty source and the safe action
# all agree, and the question templates match.
_ARCHETYPES: List[dict] = [
    {
        "scenario_clusters": ["normal_answerable_control"],
        "task_layer": "perception",
        "scenario_uncertainty": ["not_uncertain"],
        "answerability": "answerable",
        "uncertainty_source": "not_uncertain",
        "recommended_action": "proceed",
        "desc": {
            "occlusion": False,
            "sensor": False,
            "ambiguous": False,
            "multi_view": False,
            "reason": "The scene is fully observable with no notable occlusion.",
        },
        "questions": [
            {
                "question": "How many vehicles are directly ahead of the ego "
                "vehicle in its lane?",
                "target_object": "vehicle",
                "task_layer": "perception",
                "expected_answerability": "answerable",
            }
        ],
        "ground_truth_answer": "Two vehicles are visible directly ahead in the "
        "ego lane.",
        "visible_evidence": "The lane ahead and the vehicles in it are clearly "
        "visible.",
        "missing_evidence": "",
        "safety_relevance": "Ordinary controllable situation with full visibility.",
    },
    {
        "scenario_clusters": ["object_occlusion"],
        "task_layer": "perception",
        "scenario_uncertainty": ["occlusion"],
        "answerability": "unanswerable",
        "uncertainty_source": "occlusion",
        "recommended_action": "slow_down",
        "desc": {
            "occlusion": True,
            "sensor": False,
            "ambiguous": False,
            "multi_view": False,
            "reason": "A parked truck hides part of the road on the right.",
        },
        "questions": [
            {
                "question": "Is there a pedestrian stepping out from behind the "
                "parked truck on the right?",
                "target_object": "pedestrian",
                "task_layer": "perception",
                "expected_answerability": "unanswerable",
            }
        ],
        "ground_truth_answer": "Cannot determine from the available visual "
        "evidence; the area behind the truck is occluded.",
        "visible_evidence": "A parked truck occupies the right side of the view.",
        "missing_evidence": "The area behind the truck is not visible.",
        "safety_relevance": "An occluded pedestrian could appear, so the model "
        "should abstain and slow down.",
    },
    {
        "scenario_clusters": ["traffic_light_or_sign_occlusion"],
        "task_layer": "infrastructure",
        "scenario_uncertainty": ["occlusion"],
        "answerability": "unanswerable",
        "uncertainty_source": "occlusion",
        "recommended_action": "stop_or_wait",
        "desc": {
            "occlusion": True,
            "sensor": False,
            "ambiguous": False,
            "multi_view": False,
            "reason": "The traffic light controlling the ego lane is blocked by "
            "a high vehicle.",
        },
        "questions": [
            {
                "question": "What color is the traffic light controlling the ego "
                "lane?",
                "target_object": "traffic light",
                "task_layer": "infrastructure",
                "expected_answerability": "unanswerable",
            }
        ],
        "ground_truth_answer": "Cannot determine; the relevant traffic light is "
        "not visible.",
        "visible_evidence": "An intersection with a tall vehicle ahead is visible.",
        "missing_evidence": "The traffic light state is hidden behind the vehicle.",
        "safety_relevance": "Acting on a guessed light state is unsafe; the model "
        "should abstain and wait.",
    },
    {
        "scenario_clusters": ["sensor_degradation"],
        "task_layer": "environment",
        "scenario_uncertainty": ["sensor_degradation"],
        "answerability": "unanswerable",
        "uncertainty_source": "sensor_degradation",
        "recommended_action": "slow_down",
        "desc": {
            "occlusion": False,
            "sensor": True,
            "ambiguous": False,
            "multi_view": False,
            "reason": "Lens glare and motion blur degrade the forward image.",
        },
        "questions": [
            {
                "question": "What does the road sign on the right say?",
                "target_object": "road sign",
                "task_layer": "infrastructure",
                "expected_answerability": "unanswerable",
            }
        ],
        "ground_truth_answer": "Cannot determine; image degradation makes the "
        "sign illegible.",
        "visible_evidence": "A sign-shaped object is visible on the right.",
        "missing_evidence": "Glare and blur make the sign text unreadable.",
        "safety_relevance": "Degraded sensors reduce confidence, so the model "
        "should abstain and slow down.",
    },
    {
        "scenario_clusters": ["ambiguous_agent_intent"],
        "task_layer": "prediction",
        "scenario_uncertainty": ["future_intent_unknown"],
        "answerability": "ambiguous",
        "uncertainty_source": "future_intent_unknown",
        "recommended_action": "slow_down",
        "desc": {
            "occlusion": False,
            "sensor": False,
            "ambiguous": True,
            "multi_view": False,
            "reason": "A pedestrian near the curb may or may not be about to cross.",
        },
        "questions": [
            {
                "question": "Will the pedestrian at the curb cross in front of the "
                "ego vehicle?",
                "target_object": "pedestrian",
                "task_layer": "prediction",
                "expected_answerability": "ambiguous",
            }
        ],
        "ground_truth_answer": "Ambiguous; the pedestrian's intent to cross cannot "
        "be determined from a single frame.",
        "visible_evidence": "A pedestrian is standing near the curb edge.",
        "missing_evidence": "The pedestrian's future intent is unknowable here.",
        "safety_relevance": "Intent is ambiguous, so the model should abstain and "
        "slow down.",
    },
    {
        "scenario_clusters": ["planning_under_occlusion"],
        "task_layer": "planning",
        "scenario_uncertainty": ["occlusion"],
        "answerability": "ambiguous",
        "uncertainty_source": "occlusion",
        "recommended_action": "stop_or_wait",
        "desc": {
            "occlusion": True,
            "sensor": False,
            "ambiguous": True,
            "multi_view": False,
            "reason": "An occluded side road makes a turn decision uncertain.",
        },
        "questions": [
            {
                "question": "Is it safe for the ego vehicle to turn right now?",
                "target_object": "side road",
                "task_layer": "planning",
                "expected_answerability": "ambiguous",
            }
        ],
        "ground_truth_answer": "Ambiguous; oncoming traffic from the occluded side "
        "road cannot be confirmed.",
        "visible_evidence": "A right-turn opportunity is visible.",
        "missing_evidence": "The occluded side road may hide oncoming traffic.",
        "safety_relevance": "Planning under occlusion is risky, so the model "
        "should abstain and wait.",
    },
    {
        "scenario_clusters": ["risk_under_incomplete_evidence"],
        "task_layer": "planning",
        "scenario_uncertainty": ["conflicting_visual_evidence"],
        "answerability": "unanswerable",
        "uncertainty_source": "conflicting_visual_evidence",
        "recommended_action": "minimal_risk_response",
        "desc": {
            "occlusion": True,
            "sensor": True,
            "ambiguous": False,
            "multi_view": False,
            "reason": "Conflicting cues and partial visibility make the situation "
            "high-risk.",
        },
        "questions": [
            {
                "question": "Can the ego vehicle proceed through the intersection "
                "safely?",
                "target_object": "intersection",
                "task_layer": "planning",
                "expected_answerability": "unanswerable",
            }
        ],
        "ground_truth_answer": "Cannot determine safely; visual evidence is "
        "incomplete and conflicting.",
        "visible_evidence": "A complex intersection with partial obstructions.",
        "missing_evidence": "Key cross-traffic and signal evidence is missing.",
        "safety_relevance": "High risk under incomplete evidence warrants a "
        "minimal-risk response.",
    },
    {
        "scenario_clusters": ["multi_view_required"],
        "task_layer": "perception",
        "scenario_uncertainty": ["multi_view_missing"],
        "answerability": "unanswerable",
        "uncertainty_source": "multi_view_missing",
        "recommended_action": "slow_down",
        "desc": {
            "occlusion": False,
            "sensor": False,
            "ambiguous": False,
            "multi_view": True,
            "reason": "Answering requires a camera view that is not available.",
        },
        "questions": [
            {
                "question": "Is a vehicle approaching in the ego vehicle's blind "
                "spot on the left?",
                "target_object": "vehicle",
                "task_layer": "perception",
                "expected_answerability": "unanswerable",
            }
        ],
        "ground_truth_answer": "Cannot determine; the required side/blind-spot view "
        "is not available.",
        "visible_evidence": "Front views are clear but the left blind spot is not "
        "covered.",
        "missing_evidence": "The left blind-spot camera view is missing.",
        "safety_relevance": "Without the needed view the model should abstain and "
        "slow down.",
    },
]

# A second, optional follow-up question used for some samples so question
# counts vary between one and two.
_SECOND_QUESTION = {
    "question": "Are there any cyclists visible to the side of the ego vehicle?",
    "target_object": "cyclist",
    "task_layer": "perception",
    "expected_answerability": "answerable",
}


def _sample_seed(image_paths: List[str]) -> int:
    """Derive a stable integer seed from the sorted preview file names."""
    basenames = sorted(Path(path).name for path in image_paths if path)
    digest = short_hash("|".join(basenames) or "no-images", 16)
    return int(digest, 16)


def _archetype_for(seed: int) -> dict:
    return _ARCHETYPES[seed % len(_ARCHETYPES)]


def _is_not_relevant(seed: int) -> bool:
    # Roughly one sample in 13 is labelled not relevant.
    return seed % 13 == 0


def _build_description(archetype: dict) -> dict:
    desc = archetype["desc"]
    return {
        "street": {
            "road_layout": "Urban road segment as seen from the ego vehicle.",
            "lanes_visible": "Multiple forward lanes are visible.",
            "crosswalk_visible": "A crosswalk is visible ahead.",
            "occluded_regions": (
                ["Region hidden behind a nearby large object"]
                if desc["occlusion"]
                else []
            ),
        },
        "infrastructure": {
            "traffic_lights": ["Traffic light ahead"],
            "traffic_signs": ["Standard regulatory signage"],
            "barriers_or_construction": [],
            "visibility_issues": (
                ["A traffic light or sign is partially occluded"]
                if desc["occlusion"]
                else []
            ),
        },
        "movable_objects": {
            "vehicles": ["Cars ahead in the ego lane"],
            "pedestrians": (
                ["Pedestrian near the curb"] if desc["ambiguous"] else []
            ),
            "cyclists": [],
            "ambiguous_intentions": (
                ["Pedestrian intent to cross is unclear"]
                if desc["ambiguous"]
                else []
            ),
        },
        "environment": {
            "weather": "Clear",
            "lighting": "Daylight",
            "visibility": "Reduced" if desc["sensor"] else "Good",
            "image_quality_issues": (
                ["Glare and motion blur"] if desc["sensor"] else []
            ),
        },
        "uncertainty": {
            "possible_occlusion": desc["occlusion"],
            "possible_sensor_degradation": desc["sensor"],
            "possible_ambiguous_intent": desc["ambiguous"],
            "multi_view_needed": desc["multi_view"],
            "uncertainty_reason": desc["reason"],
        },
    }


def _build_scenario(archetype: dict, seed: int) -> dict:
    if _is_not_relevant(seed):
        return {
            "is_relevant": False,
            "scenario_clusters": ["not_relevant"],
            "task_layer": "perception",
            "uncertainty_sources": ["not_uncertain"],
            "safety_relevance": "Ordinary, fully observable scene with no edge case.",
            "reason": "Clear scene with full visibility and no uncertainty of "
            "interest.",
        }
    return {
        "is_relevant": True,
        "scenario_clusters": list(archetype["scenario_clusters"]),
        "task_layer": archetype["task_layer"],
        "uncertainty_sources": list(archetype["scenario_uncertainty"]),
        "safety_relevance": archetype["safety_relevance"],
        "reason": archetype["desc"]["reason"],
    }


def _build_questions(archetype: dict, seed: int) -> dict:
    questions = [dict(archetype["questions"][0])]
    # Add a second question for roughly half the samples.
    if seed % 2 == 0:
        questions.append(dict(_SECOND_QUESTION))
    return {"questions": questions}


def _build_answerability(archetype: dict, prompt: str) -> dict:
    # The generic follow-up question is always answerable. Match its exact
    # text (not just "cyclist", which also appears as a description JSON key).
    if _SECOND_QUESTION["question"] in prompt:
        return {
            "answerability": "answerable",
            "ground_truth_answer": "No cyclists are visible beside the ego vehicle.",
            "abstention_required": False,
            "visible_evidence": "The side areas beside the ego vehicle are visible.",
            "missing_evidence": "",
            "uncertainty_source": "not_uncertain",
            "recommended_action": "proceed",
            "rationale": "The side areas are clearly visible, so the question is "
            "answerable.",
        }
    answerability = archetype["answerability"]
    return {
        "answerability": answerability,
        "ground_truth_answer": archetype["ground_truth_answer"],
        "abstention_required": answerability in {"unanswerable", "ambiguous"},
        "visible_evidence": archetype["visible_evidence"],
        "missing_evidence": archetype["missing_evidence"],
        "uncertainty_source": archetype["uncertainty_source"],
        "recommended_action": archetype["recommended_action"],
        "rationale": archetype["safety_relevance"],
    }


class MockGeminiClient(BaseLLMClient):
    """Mock client returning schema-conforming canned JSON (no API calls)."""

    client_kind = "mock"

    def _check_call_budget(self) -> None:
        # Mock calls cost nothing, so the paid-call budget does not apply.
        return

    def _produce_raw_text(
        self, prompt: str, schema: dict, image_paths: List[str]
    ) -> str:
        title = (schema or {}).get("title")
        seed = _sample_seed(image_paths)
        archetype = _archetype_for(seed)

        if title == "LayeredSceneDescription":
            data: dict = _build_description(archetype)
        elif title == "ScenarioClassification":
            data = _build_scenario(archetype, seed)
        elif title == "QuestionList":
            data = _build_questions(archetype, seed)
        elif title == "AnswerabilityLabel":
            data = _build_answerability(archetype, prompt)
        else:
            raise ValueError(
                f"MockGeminiClient cannot handle schema title {title!r}."
            )

        return json.dumps(data)
