"""Image loading, resizing and grid-composition helpers (Pillow)."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from PIL import Image, ImageDraw

# Mime type lookup for the image formats we emit.
_MIME_BY_SUFFIX = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
}


def mime_type_for(path: str | Path) -> str:
    """Return the image mime type for a path based on its suffix."""
    suffix = Path(path).suffix.lower()
    return _MIME_BY_SUFFIX.get(suffix, "image/jpeg")


def load_image(path: str | Path) -> Image.Image:
    """Open an image and convert it to RGB."""
    with Image.open(path) as img:
        return img.convert("RGB")


def resize_max_side(image: Image.Image, max_side_pixels: int) -> Image.Image:
    """Downscale so the longest side equals `max_side_pixels` (no upscaling)."""
    width, height = image.size
    longest = max(width, height)
    if longest <= max_side_pixels:
        return image
    scale = max_side_pixels / float(longest)
    new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return image.resize(new_size, Image.LANCZOS)


def save_image(image: Image.Image, path: str | Path, quality: int = 90) -> None:
    """Save an image, creating parent directories as needed."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix.lower() in (".jpg", ".jpeg"):
        image.save(out, format="JPEG", quality=quality)
    else:
        image.save(out)


def make_grid(
    images: List[Optional[Image.Image]],
    cols: int,
    rows: int,
    labels: Optional[List[str]] = None,
    cell_size: tuple[int, int] = (320, 180),
    background: tuple[int, int, int] = (0, 0, 0),
) -> Image.Image:
    """Compose images into a `cols` x `rows` grid.

    ``None`` entries render as labelled black tiles so missing camera
    views remain visible in the layout.
    """
    cell_w, cell_h = cell_size
    grid = Image.new("RGB", (cols * cell_w, rows * cell_h), background)
    draw = ImageDraw.Draw(grid)

    for index in range(cols * rows):
        row, col = divmod(index, cols)
        x0, y0 = col * cell_w, row * cell_h
        source = images[index] if index < len(images) else None
        if source is not None:
            tile = source.copy()
            tile.thumbnail((cell_w, cell_h), Image.LANCZOS)
            offset_x = x0 + (cell_w - tile.width) // 2
            offset_y = y0 + (cell_h - tile.height) // 2
            grid.paste(tile, (offset_x, offset_y))
        if labels and index < len(labels) and labels[index]:
            draw.text((x0 + 4, y0 + 4), labels[index], fill=(255, 255, 0))

    return grid
