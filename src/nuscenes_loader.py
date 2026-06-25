"""Clean wrapper around the nuScenes devkit (Phase 1).

This module performs no Gemini calls. The heavy ``nuscenes`` import is
done lazily inside the constructor so that other stages (and the unit
tests) can import this module without the dataset being present.
"""

from __future__ import annotations

import os
from typing import Dict, Iterator, List, Optional

from constants import CAMERA_NAMES
from utils.logging_utils import get_logger

logger = get_logger(__name__)


class NuScenesLoader:
    """Thin convenience layer over :class:`nuscenes.nuscenes.NuScenes`."""

    def __init__(self, dataroot: str, version: str, verbose: bool = False):
        if not os.path.isdir(dataroot):
            raise FileNotFoundError(
                f"nuScenes dataroot does not exist: {dataroot}. "
                f"Download the dataset or fix 'dataset.dataroot' in the config."
            )
        # Imported lazily: the devkit pulls in a heavy scientific stack.
        from nuscenes.nuscenes import NuScenes

        logger.info("Loading nuScenes version=%s dataroot=%s", version, dataroot)
        self.dataroot = dataroot
        self.version = version
        self.nusc = NuScenes(version=version, dataroot=dataroot, verbose=verbose)

    def iter_samples(self) -> Iterator[dict]:
        """Yield every sample record in the dataset."""
        for sample in self.nusc.sample:
            yield sample

    def get_sample(self, sample_token: str) -> dict:
        """Return a single sample record."""
        return self.nusc.get("sample", sample_token)

    def get_scene(self, scene_token: str) -> dict:
        """Return a single scene record."""
        return self.nusc.get("scene", scene_token)

    def get_camera_paths(self, sample_token: str) -> Dict[str, Optional[str]]:
        """Return absolute image paths for each camera.

        Missing files (or cameras absent from the sample) map to ``None``
        so downstream stages can record their missing status.
        """
        sample = self.get_sample(sample_token)
        paths: Dict[str, Optional[str]] = {}
        for camera in CAMERA_NAMES:
            sd_token = sample["data"].get(camera)
            if sd_token is None:
                paths[camera] = None
                continue
            path = self.nusc.get_sample_data_path(sd_token)
            if os.path.exists(path):
                paths[camera] = path
            else:
                logger.warning("Missing image file for %s: %s", camera, path)
                paths[camera] = None
        return paths

    def get_annotations(self, sample_token: str) -> List[dict]:
        """Return all sample-annotation records for a sample."""
        sample = self.get_sample(sample_token)
        return [self.nusc.get("sample_annotation", token) for token in sample["anns"]]

    def get_ego_pose(self, sample_token: str) -> dict:
        """Return the ego pose associated with the front camera frame."""
        sample = self.get_sample(sample_token)
        reference = sample["data"].get("CAM_FRONT") or next(
            iter(sample["data"].values())
        )
        sample_data = self.nusc.get("sample_data", reference)
        return self.nusc.get("ego_pose", sample_data["ego_pose_token"])
