"""Annotation report generation (Phase 18).

Reads the auto-candidate, verified and rejected JSONL files plus the
sample index, computes distributions and writes both a machine-readable
JSON summary and a human-readable Markdown report.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Dict, List

from config import AppConfig
from storage import load_jsonl_if_exists, save_json
from utils.logging_utils import get_logger

logger = get_logger(__name__)


def _distinct_samples(records: List[dict]) -> int:
    return len({r.get("sample_id") for r in records})


def _scenario_clusters(record: dict) -> List[str]:
    scenario = record.get("scenario_classification") or {}
    return scenario.get("scenario_clusters") or []


def _label_field(record: dict, field: str) -> str:
    label = record.get("answerability_label") or {}
    return label.get(field, "")


def build_summary(
    indexed: List[dict],
    candidates: List[dict],
    verified: List[dict],
    rejected: List[dict],
) -> Dict:
    """Compute the annotation-summary dictionary from loaded records."""
    cluster_counts: Counter = Counter()
    for record in candidates:
        cluster_counts.update(_scenario_clusters(record))

    answerability_counts = Counter(
        _label_field(r, "answerability") for r in candidates
    )
    action_counts = Counter(
        _label_field(r, "recommended_action") for r in candidates
    )
    uncertainty_counts = Counter(
        _label_field(r, "uncertainty_source") for r in candidates
    )

    # One example verified record per scenario cluster (fall back to
    # candidates when no verified samples exist yet).
    example_source = verified if verified else candidates
    examples_by_cluster: Dict[str, dict] = {}
    for record in example_source:
        for cluster in _scenario_clusters(record):
            if cluster not in examples_by_cluster:
                examples_by_cluster[cluster] = {
                    "sample_id": record.get("sample_id"),
                    "question": record.get("question"),
                    "answerability": _label_field(record, "answerability"),
                    "ground_truth_answer": _label_field(
                        record, "ground_truth_answer"
                    ),
                    "recommended_action": _label_field(
                        record, "recommended_action"
                    ),
                }

    return {
        "num_indexed_samples": len(indexed),
        "num_gemini_processed_samples": _distinct_samples(candidates),
        "num_candidate_records": len(candidates),
        "num_verified_samples": len(verified),
        "num_rejected_samples": len(rejected),
        "scenario_cluster_distribution": dict(cluster_counts),
        "answerability_distribution": dict(answerability_counts),
        "safety_action_distribution": dict(action_counts),
        "uncertainty_source_distribution": dict(uncertainty_counts),
        "examples_used_from": "verified" if verified else "candidates",
        "example_records_per_cluster": examples_by_cluster,
    }


def _render_distribution(title: str, counts: Dict[str, int]) -> str:
    if not counts:
        return f"### {title}\n\n_None._\n"
    lines = [f"### {title}\n", "| Value | Count |", "| --- | --- |"]
    for key, value in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
        lines.append(f"| {key} | {value} |")
    return "\n".join(lines) + "\n"


def render_markdown(summary: Dict) -> str:
    """Render the summary dictionary as a Markdown report."""
    parts: List[str] = ["# Annotation Summary\n"]
    parts.append("## Counts\n")
    parts.append(
        "\n".join(
            [
                "| Metric | Value |",
                "| --- | --- |",
                f"| Indexed samples | {summary['num_indexed_samples']} |",
                f"| Gemini processed samples | {summary['num_gemini_processed_samples']} |",
                f"| Candidate records | {summary['num_candidate_records']} |",
                f"| Verified samples | {summary['num_verified_samples']} |",
                f"| Rejected samples | {summary['num_rejected_samples']} |",
            ]
        )
        + "\n"
    )

    parts.append("## Distributions\n")
    parts.append(
        _render_distribution(
            "Scenario clusters", summary["scenario_cluster_distribution"]
        )
    )
    parts.append(
        _render_distribution(
            "Answerability", summary["answerability_distribution"]
        )
    )
    parts.append(
        _render_distribution(
            "Safety actions", summary["safety_action_distribution"]
        )
    )
    parts.append(
        _render_distribution(
            "Uncertainty sources", summary["uncertainty_source_distribution"]
        )
    )

    parts.append(
        f"## Example records per cluster (from {summary['examples_used_from']})\n"
    )
    examples = summary["example_records_per_cluster"]
    if not examples:
        parts.append("_No examples available._\n")
    else:
        for cluster, example in sorted(examples.items()):
            parts.append(
                f"- **{cluster}** — sample `{example['sample_id']}`: "
                f"\"{example['question']}\" "
                f"({example['answerability']}, action: "
                f"{example['recommended_action']})\n"
                f"  - Answer: {example['ground_truth_answer']}"
            )
        parts.append("")

    return "\n".join(parts)


def generate_report(config: AppConfig) -> Dict:
    """Generate the JSON + Markdown annotation report and return the summary."""
    indexed = load_jsonl_if_exists(config.paths.sample_index_path)
    candidates = load_jsonl_if_exists(config.paths.auto_candidates_path)
    verified = load_jsonl_if_exists(config.paths.verified_output_path)
    rejected = load_jsonl_if_exists(config.paths.rejected_output_path)

    summary = build_summary(indexed, candidates, verified, rejected)

    report_dir = Path(config.paths.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    json_path = report_dir / "annotation_summary.json"
    md_path = report_dir / "annotation_summary.md"

    save_json(summary, json_path)
    with open(md_path, "w", encoding="utf-8") as handle:
        handle.write(render_markdown(summary))

    logger.info("Wrote report to %s and %s", json_path, md_path)
    return summary
