"""Tests for JSON/JSONL storage helpers (Phase 0 acceptance criteria)."""

from schema import SceneIndexRecord
from storage import (
    append_jsonl,
    load_json,
    load_jsonl,
    load_jsonl_if_exists,
    save_json,
    save_jsonl,
)


def test_json_round_trip(tmp_path):
    path = tmp_path / "obj.json"
    data = {"a": 1, "b": ["x", "y"], "c": {"nested": True}}
    save_json(data, path)
    assert load_json(path) == data


def test_jsonl_round_trip(tmp_path):
    path = tmp_path / "records.jsonl"
    records = [{"id": 1}, {"id": 2}, {"id": 3}]
    save_jsonl(records, path)
    assert load_jsonl(path) == records


def test_jsonl_accepts_pydantic_models(tmp_path):
    path = tmp_path / "scenes.jsonl"
    record = SceneIndexRecord(
        sample_id="s1",
        scene_token="sc1",
        timestamp=123,
        camera_paths={"CAM_FRONT": "front.jpg"},
        num_annotations=2,
        annotation_categories=["vehicle.car"],
        category_counts={"vehicle.car": 2},
        scene_description="A street scene.",
    )
    save_jsonl([record], path)
    loaded = load_jsonl(path)
    assert loaded[0]["sample_id"] == "s1"
    # Round-trips back into a model.
    assert SceneIndexRecord.model_validate(loaded[0]) == record


def test_append_jsonl(tmp_path):
    path = tmp_path / "appended.jsonl"
    append_jsonl({"id": 1}, path)
    append_jsonl({"id": 2}, path)
    assert load_jsonl(path) == [{"id": 1}, {"id": 2}]


def test_load_jsonl_if_exists_missing_returns_empty(tmp_path):
    assert load_jsonl_if_exists(tmp_path / "nope.jsonl") == []


def test_unicode_is_preserved(tmp_path):
    path = tmp_path / "unicode.jsonl"
    save_jsonl([{"text": "Fußgänger über Straße"}], path)
    assert load_jsonl(path)[0]["text"] == "Fußgänger über Straße"
