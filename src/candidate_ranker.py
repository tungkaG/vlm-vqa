"""Candidate ranking (Phase 14).

Assigns a priority score to each candidate so the verification GUI shows
the most useful (uncertainty- and safety-relevant) samples first,
removes duplicate sample/question pairs, and sorts by score.
"""

from __future__ import annotations

from typing import List

from schema import CandidateRecord
from storage import load_jsonl, save_jsonl
from utils.logging_utils import get_logger

logger = get_logger(__name__)


def compute_priority(candidate: CandidateRecord) -> float:
    """Score a candidate by uncertainty and safety relevance (README rules)."""
    score = 0.0
    label = candidate.answerability_label

    if label.answerability == "unanswerable":
        score += 4
    if label.answerability == "ambiguous":
        score += 3

    if label.recommended_action == "slow_down":
        score += 2
    if label.recommended_action == "stop_or_wait":
        score += 3
    if label.recommended_action == "minimal_risk_response":
        score += 4

    clusters = candidate.scenario_classification.scenario_clusters
    if "multi_view_required" in clusters:
        score += 2
    if "normal_answerable_control" in clusters:
        score += 1

    return float(score)


def deduplicate_candidates(
    candidates: List[CandidateRecord],
) -> List[CandidateRecord]:
    """Drop duplicate (sample_id, question) pairs, keeping the first."""
    seen: set[tuple[str, str]] = set()
    unique: List[CandidateRecord] = []
    for candidate in candidates:
        key = (candidate.sample_id, candidate.question.strip().lower())
        if key in seen:
            continue
        seen.add(key)
        unique.append(candidate)
    return unique


def sort_candidates_by_priority(
    candidates: List[CandidateRecord],
) -> List[CandidateRecord]:
    """Return candidates sorted by descending priority score (stable)."""
    return sorted(candidates, key=lambda c: c.priority_score, reverse=True)


def rank_candidates(
    candidates: List[CandidateRecord],
) -> List[CandidateRecord]:
    """Score, deduplicate and sort candidates by priority."""
    for candidate in candidates:
        candidate.priority_score = compute_priority(candidate)
    unique = deduplicate_candidates(candidates)
    return sort_candidates_by_priority(unique)


def rank_candidates_file(input_path: str, output_path: str) -> List[CandidateRecord]:
    """Load candidates, rank them and save the ranked JSONL."""
    rows = load_jsonl(input_path)
    candidates = [CandidateRecord.model_validate(row) for row in rows]
    ranked = rank_candidates(candidates)
    save_jsonl(ranked, output_path)
    logger.info("Ranked %d candidates -> %s", len(ranked), output_path)
    return ranked
