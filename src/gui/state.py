"""GUI state and persistence logic (Phase 17 / Milestone 9).

Pure, Streamlit-free helpers for loading ranked candidates, tracking
which have been verified or rejected, merging human edits into a
candidate and appending the result to the verified/rejected JSONL files.
Keeping this logic here makes it unit-testable without a running GUI.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

from config import AppConfig
from schema import CandidateRecord
from storage import append_jsonl, load_jsonl_if_exists

# Fields a human may edit before accepting a candidate (README Phase 17).
HUMAN_EDITABLE_FIELDS = [
    "scenario_cluster",
    "task_layer",
    "question",
    "answerability",
    "ground_truth_answer",
    "abstention_required",
    "visible_evidence",
    "missing_evidence",
    "uncertainty_source",
    "recommended_action",
    "human_notes",
]


def candidate_key(record) -> Tuple[str, str]:
    """Stable (sample_id, question) identity for a candidate or dict."""
    if hasattr(record, "sample_id"):
        return (record.sample_id, (record.question or "").strip().lower())
    return (
        record.get("sample_id", ""),
        (record.get("question") or "").strip().lower(),
    )


def load_candidates(config: AppConfig) -> List[CandidateRecord]:
    """Load ranked candidate records from the auto-candidates JSONL."""
    rows = load_jsonl_if_exists(config.paths.auto_candidates_path)
    return [CandidateRecord.model_validate(row) for row in rows]


def load_processed_keys(config: AppConfig) -> Set[Tuple[str, str]]:
    """Return the keys of every already verified or rejected candidate."""
    verified, rejected = load_status_keys(config)
    return verified | rejected


def load_status_keys(
    config: AppConfig,
) -> Tuple[Set[Tuple[str, str]], Set[Tuple[str, str]]]:
    """Return (verified_keys, rejected_keys) as separate sets."""
    verified = {
        candidate_key(row)
        for row in load_jsonl_if_exists(config.paths.verified_output_path)
    }
    rejected = {
        candidate_key(row)
        for row in load_jsonl_if_exists(config.paths.rejected_output_path)
    }
    return verified, rejected



def count_verified(config: AppConfig) -> int:
    """Return the number of verified samples saved so far."""
    return len(load_jsonl_if_exists(config.paths.verified_output_path))


def is_target_reached(config: AppConfig) -> bool:
    """True when the verified count meets the configured target."""
    return count_verified(config) >= config.pipeline.target_verified_count


def select_next_candidate(
    candidates: List[CandidateRecord],
    processed_keys: Set[Tuple[str, str]],
) -> Optional[CandidateRecord]:
    """Return the highest-priority candidate not yet verified or rejected."""
    for candidate in candidates:
        if candidate_key(candidate) not in processed_keys:
            return candidate
    return None


def build_verified_record(
    candidate: CandidateRecord, edits: Dict
) -> CandidateRecord:
    """Merge human edits into a candidate and re-validate it.

    Re-validation enforces every controlled vocabulary and the
    abstention-consistency rule, so an inconsistent human edit raises a
    clear ``ValidationError``.
    """
    data = candidate.model_dump()

    scenario = data["scenario_classification"]
    scenario["scenario_clusters"] = [edits["scenario_cluster"]]
    scenario["task_layer"] = edits["task_layer"]

    label = data["answerability_label"]
    label["answerability"] = edits["answerability"]
    label["ground_truth_answer"] = edits["ground_truth_answer"]
    label["abstention_required"] = edits["abstention_required"]
    label["visible_evidence"] = edits["visible_evidence"]
    label["missing_evidence"] = edits["missing_evidence"]
    label["uncertainty_source"] = edits["uncertainty_source"]
    label["recommended_action"] = edits["recommended_action"]

    data["question"] = edits["question"]
    data["human_notes"] = edits.get("human_notes") or None
    data["human_verified"] = True

    return CandidateRecord.model_validate(data)


def accept_candidate(
    candidate: CandidateRecord, edits: Dict, config: AppConfig
) -> CandidateRecord:
    """Build the verified record and append it to the verified JSONL."""
    record = build_verified_record(candidate, edits)
    append_jsonl(record, config.paths.verified_output_path)
    return record


def reject_candidate(
    candidate: CandidateRecord, config: AppConfig, notes: Optional[str] = None
) -> None:
    """Append a candidate to the rejected JSONL (optionally with notes)."""
    data = candidate.model_dump()
    if notes:
        data["human_notes"] = notes
    append_jsonl(data, config.paths.rejected_output_path)
