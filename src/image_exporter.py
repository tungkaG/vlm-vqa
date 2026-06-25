"""Camera preview and multi-view grid generation (Phase 3).

Resizes nuScenes camera images for Gemini request-size control and the
GUI, caching by source-file hash so re-runs skip work. Original image
paths are preserved by the caller (the index keeps ``camera_paths``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from constants import CAMERA_NAMES
from utils.hash_utils import short_hash
from utils.image_utils import load_image, make_grid, resize_max_side, save_image
from utils.logging_utils import get_logger

logger = get_logger(__name__)

# Layout for the six-camera grid: front row on top, back row below.
_GRID_LAYOUT = [
    "CAM_FRONT_LEFT",
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK_LEFT",
    "CAM_BACK",
    "CAM_BACK_RIGHT",
]


def create_camera_previews(
    camera_paths: Dict[str, Optional[str]],
    output_dir: str,
    max_side_pixels: int,
) -> Dict[str, Optional[str]]:
    """Create resized preview images for each available camera.

    Returns a mapping of camera name to preview path (``None`` when the
    source image is missing). Existing previews are reused (cached).
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    previews: Dict[str, Optional[str]] = {}
    for camera in CAMERA_NAMES:
        source = camera_paths.get(camera)
        if not source:
            previews[camera] = None
            continue

        preview_path = out_dir / f"{camera}_{short_hash(source)}.jpg"
        if preview_path.exists():
            previews[camera] = str(preview_path)
            continue

        try:
            image = load_image(source)
            image = resize_max_side(image, max_side_pixels)
            save_image(image, preview_path)
            previews[camera] = str(preview_path)
        except Exception as error:  # noqa: BLE001 - log and mark missing
            logger.warning("Failed to create preview for %s: %s", source, error)
            previews[camera] = None

    return previews


def create_multiview_grid(
    preview_paths: Dict[str, Optional[str]],
    output_path: str,
) -> str:
    """Compose the six camera previews into a single grid image."""
    images = []
    labels: List[str] = []
    for camera in _GRID_LAYOUT:
        path = preview_paths.get(camera)
        images.append(load_image(path) if path else None)
        labels.append(camera)

    grid = make_grid(images, cols=3, rows=2, labels=labels)
    save_image(grid, output_path)
    logger.info("Wrote multi-view grid to %s", output_path)
    return str(output_path)
