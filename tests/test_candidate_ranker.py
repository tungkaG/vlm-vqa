"""Tests for candidate ranking (Phase 14).

Pure local logic; no Gemini calls and no mock client.
"""

from schema import (
    AnswerabilityLabel,
    CandidateRecord,
    LayeredSceneDescription,
    ScenarioClassification,
)
from candidate_ranker import (
    compute_priority,
    deduplicate_candidates,
    rank_candidates,
    rank_candidates_file,
    sort_candidates_by_priority,
)


def _label(answerability: str, action: str, uncertainty_source: str) -> AnswerabilityLabel:
    return AnswerabilityLabel(
        answerability=answerability,
        ground_truth_answer="An answer.",
        abstention_required=answerability in {"unanswerable", "ambiguous"},
        uncertainty_source=uncertainty_source,
        recommended_action=action,
    )


def _description() -> LayeredSceneDescription:
    return LayeredSceneDescription.model_validate(
        {
            "street": {},
            "infrastructure": {},
            "movable_objects": {},
            "environment": {},
            "uncertainty": {},
        }
    )


def _candidate(
    sample_id: str,
    question: str,
    *,
    answerability: str = "answerable",
    action: str = "proceed",
    clusters=None,
    uncertainty_source: str = "not_uncertain",
) -> CandidateRecord:
    return CandidateRecord(
        sample_id=sample_id,
        scene_token="scene",
        camera_paths={"CAM_FRONT": "a.jpg"},
        preview_paths={"CAM_FRONT": "p.jpg"},
        layered_scene_description=_description(),
        scenario_classification=ScenarioClassification(
            is_relevant=True,
            scenario_clusters=clusters or ["normal_answerable_control"],
            task_layer="perception",
            uncertainty_sources=[],
        ),
        question=question,
        answerability_label=_label(answerability, action, uncertainty_source),
        source_model_name="gemini-2.5-flash",
    )


def test_compute_priority_scoring_rules():
    normal = _candidate("s1", "q", clusters=["normal_answerable_control"])
    assert compute_priority(normal) == 1.0

    occlusion = _candidate(
        "s2",
        "q",
        answerability="unanswerable",
        action="slow_down",
        clusters=["object_occlusion"],
        uncertainty_source="occlusion",
    )
    assert compute_priority(occlusion) == 6.0  # 4 + 2

    ambiguous = _candidate(
        "s3",
        "q",
        answerability="ambiguous",
        action="stop_or_wait",
        clusters=["planning_under_occlusion"],
        uncertainty_source="occlusion",
    )
    assert compute_priority(ambiguous) == 6.0  # 3 + 3

    highest = _candidate(
        "s4",
        "q",
        answerability="unanswerable",
        action="minimal_risk_response",
        clusters=["multi_view_required"],
        uncertainty_source="multi_view_missing",
    )
    assert compute_priority(highest) == 10.0  # 4 + 4 + 2


def test_priority_is_higher_for_unanswerable_safety_critical_samples():
    normal = _candidate(
        "s1",
        "q",
        clusters=["normal_answerable_control"],
    )
    safety_critical = _candidate(
        "s2",
        "q",
        answerability="unanswerable",
        action="stop_or_wait",
        clusters=["risk_under_incomplete_evidence"],
        uncertainty_source="occlusion",
    )
    assert compute_priority(safety_critical) > compute_priority(normal)


def test_rank_sorts_by_descending_priority():
    low = _candidate("s1", "q1", clusters=["normal_answerable_control"])
    high = _candidate(
        "s2",
        "q2",
        answerability="unanswerable",
        action="minimal_risk_response",
        clusters=["multi_view_required"],
        uncertainty_source="multi_view_missing",
    )
    ranked = rank_candidates([low, high])
    assert [c.sample_id for c in ranked] == ["s2", "s1"]
    assert ranked[0].priority_score >= ranked[1].priority_score


def test_rank_removes_duplicate_sample_question_pairs():
    a = _candidate("s1", "Same question?")
    b = _candidate("s1", "same question?")  # case-insensitive duplicate
    c = _candidate("s1", "Different question?")
    ranked = rank_candidates([a, b, c])
    assert len(ranked) == 2


def test_deduplicate_keeps_distinct_samples():
    a = _candidate("s1", "q")
    b = _candidate("s2", "q")  # same question, different sample -> kept
    assert len(deduplicate_candidates([a, b])) == 2


def test_sort_is_stable_for_equal_scores():
    a = _candidate("s1", "q1")
    b = _candidate("s2", "q2")
    a.priority_score = compute_priority(a)
    b.priority_score = compute_priority(b)
    ordered = sort_candidates_by_priority([a, b])
    assert [c.sample_id for c in ordered] == ["s1", "s2"]


def test_rank_candidates_file_writes_jsonl(tmp_path):
    from storage import save_jsonl

    candidates = [
        _candidate("s1", "q1", clusters=["normal_answerable_control"]),
        _candidate(
            "s2",
            "q2",
            answerability="unanswerable",
            action="stop_or_wait",
            clusters=["traffic_light_or_sign_occlusion"],
            uncertainty_source="occlusion",
        ),
    ]
    input_path = tmp_path / "auto_candidates.jsonl"
    save_jsonl(candidates, str(input_path))

    ranked = rank_candidates_file(str(input_path), str(input_path))
    assert ranked[0].sample_id == "s2"  # higher score first
    assert all(c.priority_score >= 0 for c in ranked)

    # Reloading the written file yields valid records.
    from storage import load_jsonl

    rows = load_jsonl(str(input_path))
    assert len(rows) == 2
    assert CandidateRecord.model_validate(rows[0]).sample_id == "s2"
