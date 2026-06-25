"""Build image payload parts for the google-genai SDK (Phase 5 helper)."""

from __future__ import annotations

from typing import List

from google.genai import types

from utils.image_utils import mime_type_for
from utils.logging_utils import get_logger

logger = get_logger(__name__)


def build_image_parts(image_paths: List[str], max_images: int) -> List[types.Part]:
    """Read images from disk and return inline Gemini ``Part`` objects.

    At most ``max_images`` images are attached; extras are dropped with a
    warning so a single request never exceeds the configured budget.
    """
    selected = [p for p in image_paths if p]
    if len(selected) > max_images:
        logger.warning(
            "Got %d images but max_images_per_request=%d; truncating.",
            len(selected),
            max_images,
        )
        selected = selected[:max_images]

    parts: List[types.Part] = []
    for path in selected:
        with open(path, "rb") as handle:
            data = handle.read()
        parts.append(types.Part.from_bytes(data=data, mime_type=mime_type_for(path)))
    return parts
