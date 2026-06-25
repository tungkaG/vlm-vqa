"""CLI entry point for the nuScenes Gemini annotator.

Usage:
    python src/main.py <stage> <config_path>

Implemented stages:
    smoke_test              Validate config, env wiring and output folders.
    index                   Build/load the nuScenes sample index (Phase 2).
    preview                 Generate camera previews and grids (Phase 3).
    gemini_test             One Gemini JSON call on one image (Phase 5).
    describe                Layered scene descriptions (Phase 9).
    classify_scenarios      Scenario classification (Phase 10).
    generate_questions      Candidate questions (Phase 11).
    classify_answerability  Answerability labels (Phase 12).
    auto_annotate           Full auto-annotation pipeline (Phases 9-14).
    rank                    Rank existing auto-candidates (Phase 14).
    report                  Write the annotation summary report (Phase 18).

Gemini calls use the real client by default, or a mock client when
``gemini.mock_mode`` is true (or the GEMINI_MOCK env var is set); the
mock requires no API key and keeps the call path identical.
"""

from __future__ import annotations

import os
import sys

# Ensure the src directory is importable when run as a script.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pathlib import Path

from config import (  # noqa: E402
    AppConfig,
    get_optional_env,
    load_config,
    resolve_model_name,
)
from utils.logging_utils import get_logger  # noqa: E402

logger = get_logger("main")

# Stages defined by the README (full pipeline). Only a subset is wired up
# in Part-1 phases 0-7; the rest raise a clear "not implemented yet".
STAGES = [
    "smoke_test",
    "index",
    "preview",
    "gemini_test",
    "describe",
    "classify_scenarios",
    "generate_questions",
    "classify_answerability",
    "auto_annotate",
    "rank",
    "report",
]

_IMPLEMENTED = {
    "smoke_test",
    "index",
    "preview",
    "gemini_test",
    "describe",
    "classify_scenarios",
    "generate_questions",
    "classify_answerability",
    "auto_annotate",
    "rank",
    "report",
}


def cmd_smoke_test(config: AppConfig) -> None:
    """Validate config loading, env wiring and output directories."""
    logger.info("Config loaded successfully.")
    logger.info("Dataset: name=%s version=%s", config.dataset.name, config.dataset.version)
    logger.info("Dataroot: %s", config.dataset.dataroot)
    if not Path(config.dataset.dataroot).is_dir():
        logger.warning("Dataroot does not exist yet (download nuScenes to use it).")

    logger.info("Resolved Gemini model: %s", resolve_model_name(config))
    api_key = get_optional_env(config.gemini.api_key_env)
    if api_key:
        logger.info("Gemini API key found in env '%s'.", config.gemini.api_key_env)
    else:
        logger.warning(
            "Gemini API key NOT set (env '%s'). gemini_test will fail until it is.",
            config.gemini.api_key_env,
        )

    logger.info("Output directories are ready.")
    logger.info("Smoke test OK.")


def cmd_index(config: AppConfig) -> None:
    """Build or load the nuScenes sample index."""
    from nuscenes_loader import NuScenesLoader
    from scene_indexer import build_or_load_sample_index

    loader = NuScenesLoader(
        dataroot=config.dataset.dataroot,
        version=config.dataset.version,
    )
    records = build_or_load_sample_index(
        loader=loader,
        output_path=config.paths.sample_index_path,
        max_samples=config.pipeline.max_samples,
        sample_stride=config.pipeline.sample_stride,
    )
    logger.info("Sample index ready with %d records.", len(records))


def cmd_preview(config: AppConfig) -> None:
    """Generate camera previews and a six-camera grid for each sample."""
    from image_exporter import create_camera_previews, create_multiview_grid
    from scene_indexer import load_sample_index

    index_path = config.paths.sample_index_path
    if not Path(index_path).exists():
        raise FileNotFoundError(
            f"Sample index not found at {index_path}. Run the 'index' stage first."
        )

    records = load_sample_index(index_path)
    grid_dir = Path(config.paths.preview_dir) / "grids"
    for record in records:
        previews = create_camera_previews(
            camera_paths=record.camera_paths,
            output_dir=config.paths.preview_dir,
            max_side_pixels=config.gemini.max_image_side_pixels,
        )
        if config.gui.show_multiview_grid:
            create_multiview_grid(
                preview_paths=previews,
                output_path=str(grid_dir / f"{record.sample_id}.jpg"),
            )
    logger.info("Previews generated for %d samples.", len(records))


def cmd_gemini_test(config: AppConfig) -> None:
    """Perform one real Gemini JSON call and demonstrate caching."""
    from llm.gemini_client import build_gemini_client
    from scene_indexer import load_sample_index

    client = build_gemini_client(config)

    # Prefer a real nuScenes preview; otherwise synthesize a tiny image.
    image_path = _find_one_preview_or_synthesize(config)

    schema = {
        "type": "object",
        "required": ["scene_summary", "main_objects"],
        "properties": {
            "scene_summary": {"type": "string"},
            "main_objects": {"type": "array", "items": {"type": "string"}},
        },
    }
    prompt = (
        "You are an autonomous-driving perception assistant. Look at the "
        "image and return a JSON object with 'scene_summary' (one sentence) "
        "and 'main_objects' (a list of visible object names)."
    )
    cache_key = f"gemini_test::{config.dataset.version}::{Path(image_path).name}"

    result = client.generate_json(prompt, schema, [image_path], cache_key)
    logger.info("Gemini response: %s", result)

    # Second call should be served from cache (no extra API call).
    before = client.call_count
    client.generate_json(prompt, schema, [image_path], cache_key)
    if client.call_count == before:
        logger.info("Second call served from cache (call_count unchanged).")
    logger.info("gemini_test OK. Total API calls this run: %d", client.call_count)


def _find_one_preview_or_synthesize(config: AppConfig) -> str:
    """Return a path to one image, creating a synthetic one if needed."""
    from PIL import Image

    preview_dir = Path(config.paths.preview_dir)
    if preview_dir.exists():
        for candidate in preview_dir.glob("CAM_*.jpg"):
            return str(candidate)

    synth_dir = Path(config.gemini.cache_dir).parent / "previews"
    synth_dir.mkdir(parents=True, exist_ok=True)
    synth_path = synth_dir / "gemini_test_synthetic.jpg"
    if not synth_path.exists():
        Image.new("RGB", (640, 360), (40, 80, 120)).save(synth_path, "JPEG")
        logger.info("No previews found; created a synthetic test image.")
    return str(synth_path)


# --------------------------------------------------------------------------
# Annotator stages (Phases 9-14)
# --------------------------------------------------------------------------
def _load_index_or_fail(config: AppConfig):
    """Load the sample index, or raise if the 'index' stage has not run."""
    from scene_indexer import load_sample_index

    index_path = config.paths.sample_index_path
    if not Path(index_path).exists():
        raise FileNotFoundError(
            f"Sample index not found at {index_path}. Run the 'index' stage first."
        )
    return load_sample_index(index_path)


def create_or_load_previews(sample_record, config: AppConfig) -> dict:
    """Create (or reuse cached) camera previews for one sample."""
    from image_exporter import create_camera_previews

    return create_camera_previews(
        camera_paths=sample_record.camera_paths,
        output_dir=config.paths.preview_dir,
        max_side_pixels=config.gemini.max_image_side_pixels,
    )


def _annotator_dir(config: AppConfig) -> Path:
    """Directory for intermediate per-stage annotator artifacts."""
    directory = Path(config.paths.sample_index_path).parent / "annotator"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _describe_sample(record, client, config):
    from annotator.layered_descriptor import describe_scene_with_gemini

    previews = create_or_load_previews(record, config)
    description = describe_scene_with_gemini(record, previews, client, config)
    return previews, description


def _scenario_sample(record, client, config):
    from annotator.scenario_classifier import classify_scenario_with_gemini

    previews, description = _describe_sample(record, client, config)
    scenario = classify_scenario_with_gemini(
        record, description, previews, client, config
    )
    return previews, description, scenario


def _questions_sample(record, client, config):
    from annotator.question_generator import generate_questions_with_gemini

    previews, description, scenario = _scenario_sample(record, client, config)
    if not scenario.is_relevant or "not_relevant" in scenario.scenario_clusters:
        return previews, description, scenario, []
    questions = generate_questions_with_gemini(
        record, description, scenario, previews, client, config
    )
    return previews, description, scenario, questions


def _labels_sample(record, client, config):
    from annotator.answerability_classifier import classify_answerability_with_gemini

    previews, description, scenario, questions = _questions_sample(
        record, client, config
    )
    labels = [
        classify_answerability_with_gemini(
            record, description, scenario, question, previews, client, config
        )
        for question in questions
    ]
    return previews, description, scenario, questions, labels


def cmd_describe(config: AppConfig) -> None:
    """Generate a layered scene description for every indexed sample."""
    from llm.gemini_client import build_gemini_client
    from storage import save_jsonl

    client = build_gemini_client(config)
    records = _load_index_or_fail(config)

    rows = []
    for record in records:
        _, description = _describe_sample(record, client, config)
        rows.append(
            {"sample_id": record.sample_id, "description": description.model_dump()}
        )

    output_path = _annotator_dir(config) / "descriptions.jsonl"
    save_jsonl(rows, str(output_path))
    logger.info(
        "Described %d samples -> %s (Gemini calls: %d).",
        len(rows),
        output_path,
        client.call_count,
    )


def cmd_classify_scenarios(config: AppConfig) -> None:
    """Classify the scenario for every indexed sample."""
    from llm.gemini_client import build_gemini_client
    from storage import save_jsonl

    client = build_gemini_client(config)
    records = _load_index_or_fail(config)

    rows = []
    relevant = 0
    for record in records:
        _, _, scenario = _scenario_sample(record, client, config)
        if scenario.is_relevant and "not_relevant" not in scenario.scenario_clusters:
            relevant += 1
        rows.append(
            {"sample_id": record.sample_id, "scenario": scenario.model_dump()}
        )

    output_path = _annotator_dir(config) / "scenarios.jsonl"
    save_jsonl(rows, str(output_path))
    logger.info(
        "Classified %d samples (%d relevant) -> %s (Gemini calls: %d).",
        len(rows),
        relevant,
        output_path,
        client.call_count,
    )


def cmd_generate_questions(config: AppConfig) -> None:
    """Generate candidate questions for every relevant sample."""
    from llm.gemini_client import build_gemini_client
    from storage import save_jsonl

    client = build_gemini_client(config)
    records = _load_index_or_fail(config)

    rows = []
    total_questions = 0
    for record in records:
        _, _, _, questions = _questions_sample(record, client, config)
        if not questions:
            continue
        total_questions += len(questions)
        rows.append(
            {
                "sample_id": record.sample_id,
                "questions": [q.model_dump() for q in questions],
            }
        )

    output_path = _annotator_dir(config) / "questions.jsonl"
    save_jsonl(rows, str(output_path))
    logger.info(
        "Generated %d questions across %d samples -> %s (Gemini calls: %d).",
        total_questions,
        len(rows),
        output_path,
        client.call_count,
    )


def cmd_classify_answerability(config: AppConfig) -> None:
    """Label answerability for every generated question."""
    from llm.gemini_client import build_gemini_client
    from storage import save_jsonl

    client = build_gemini_client(config)
    records = _load_index_or_fail(config)

    rows = []
    for record in records:
        _, _, _, questions, labels = _labels_sample(record, client, config)
        for question, label in zip(questions, labels):
            rows.append(
                {
                    "sample_id": record.sample_id,
                    "question": question.question,
                    "label": label.model_dump(),
                }
            )

    output_path = _annotator_dir(config) / "answerability.jsonl"
    save_jsonl(rows, str(output_path))
    logger.info(
        "Labelled %d questions -> %s (Gemini calls: %d).",
        len(rows),
        output_path,
        client.call_count,
    )


def cmd_auto_annotate(config: AppConfig) -> None:
    """Run the full auto-annotation pipeline and rank the candidates."""
    from annotator.candidate_builder import build_candidate_records
    from candidate_ranker import compute_priority, rank_candidates
    from llm.gemini_client import build_gemini_client
    from llm.gemini_types import MaxCallsExceededError
    from storage import save_jsonl

    client = build_gemini_client(config)
    records = _load_index_or_fail(config)
    output_path = config.paths.auto_candidates_path

    all_candidates = []
    stopped_early = False
    for record in records:
        try:
            previews, description, scenario, questions, labels = _labels_sample(
                record, client, config
            )
        except MaxCallsExceededError as error:
            logger.warning("Stopping auto-annotation early: %s", error)
            stopped_early = True
            break

        if not scenario.is_relevant or "not_relevant" in scenario.scenario_clusters:
            continue
        if not questions:
            continue

        candidates = build_candidate_records(
            sample_record=record,
            preview_paths=previews,
            description=description,
            scenario=scenario,
            questions=questions,
            answerability_labels=labels,
            model_name=client.model_name,
        )
        for candidate in candidates:
            candidate.priority_score = compute_priority(candidate)
            all_candidates.append(candidate)

        # Save incrementally so the run can resume after an interruption.
        save_jsonl(all_candidates, output_path)

    ranked = rank_candidates(all_candidates)
    save_jsonl(ranked, output_path)
    logger.info(
        "Auto-annotation %s: %d candidates -> %s (Gemini calls: %d).",
        "stopped early" if stopped_early else "complete",
        len(ranked),
        output_path,
        client.call_count,
    )


def cmd_rank(config: AppConfig) -> None:
    """Rank an existing auto-candidates JSONL in place."""
    from candidate_ranker import rank_candidates_file

    path = config.paths.auto_candidates_path
    if not Path(path).exists():
        raise FileNotFoundError(
            f"No candidates found at {path}. Run the 'auto_annotate' stage first."
        )
    ranked = rank_candidates_file(path, path)
    logger.info("Ranked %d candidates -> %s.", len(ranked), path)


def cmd_report(config: AppConfig) -> None:
    """Generate the annotation summary report (JSON + Markdown)."""
    from reporting import generate_report

    summary = generate_report(config)
    logger.info(
        "Report: %d indexed, %d candidate records, %d verified, %d rejected.",
        summary["num_indexed_samples"],
        summary["num_candidate_records"],
        summary["num_verified_samples"],
        summary["num_rejected_samples"],
    )


_COMMANDS = {
    "smoke_test": cmd_smoke_test,
    "index": cmd_index,
    "preview": cmd_preview,
    "gemini_test": cmd_gemini_test,
    "describe": cmd_describe,
    "classify_scenarios": cmd_classify_scenarios,
    "generate_questions": cmd_generate_questions,
    "classify_answerability": cmd_classify_answerability,
    "auto_annotate": cmd_auto_annotate,
    "rank": cmd_rank,
    "report": cmd_report,
}


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__)
        print(f"Valid stages: {', '.join(STAGES)}")
        return 2

    stage = argv[0]
    config_path = argv[1]

    if stage not in STAGES:
        logger.error("Unknown stage '%s'. Valid stages: %s", stage, ", ".join(STAGES))
        return 2
    if stage not in _IMPLEMENTED:
        logger.error(
            "Stage '%s' is not implemented in Part-1 phases 0-7 yet.", stage
        )
        return 2

    try:
        config = load_config(config_path)
        _COMMANDS[stage](config)
    except Exception as error:  # noqa: BLE001 - surface a clean CLI error
        logger.error("Stage '%s' failed: %s", stage, error)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
