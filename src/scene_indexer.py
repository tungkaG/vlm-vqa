"""Lightweight nuScenes sample index (Phase 2).

Builds a JSONL index of samples so later stages do not need to query the
devkit repeatedly. Honours ``max_samples`` and ``sample_stride``.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import List

from nuscenes_loader import NuScenesLoader
from schema import SceneIndexRecord
from storage import load_jsonl, save_jsonl
from utils.logging_utils import get_logger

logger = get_logger(__name__)


def build_sample_index(
    loader: NuScenesLoader,
    output_path: str,
    max_samples: int,
    sample_stride: int = 1,
) -> List[SceneIndexRecord]:
    """Build the sample index and write it to ``output_path``."""
    stride = max(1, sample_stride)
    records: List[SceneIndexRecord] = []

    for position, sample in enumerate(loader.iter_samples()):
        if position % stride != 0:
            continue
        if len(records) >= max_samples:
            break

        token = sample["token"]
        camera_paths = loader.get_camera_paths(token)
        annotations = loader.get_annotations(token)
        categories = [ann["category_name"] for ann in annotations]
        counts = Counter(categories)

        scene = loader.get_scene(sample["scene_token"])

        records.append(
            SceneIndexRecord(
                sample_id=token,
                scene_token=sample["scene_token"],
                timestamp=sample["timestamp"],
                camera_paths=camera_paths,
                num_annotations=len(annotations),
                annotation_categories=sorted(counts.keys()),
                category_counts=dict(counts),
                scene_description=scene.get("description"),
            )
        )

    save_jsonl(records, output_path)
    logger.info("Wrote %d sample index records to %s", len(records), output_path)
    return records


def load_sample_index(path: str) -> List[SceneIndexRecord]:
    """Load the sample index from a JSONL file."""
    return [SceneIndexRecord.model_validate(row) for row in load_jsonl(path)]


def build_or_load_sample_index(
    loader: NuScenesLoader,
    output_path: str,
    max_samples: int,
    sample_stride: int = 1,
    force: bool = False,
) -> List[SceneIndexRecord]:
    """Load an existing index, or build it if missing (or ``force``)."""
    if not force and Path(output_path).exists():
        records = load_sample_index(output_path)
        logger.info("Loaded %d cached sample index records", len(records))
        return records
    return build_sample_index(loader, output_path, max_samples, sample_stride)
