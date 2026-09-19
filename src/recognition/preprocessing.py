"""Image loading and model-owned preprocessing helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable


def load_rgb_image(image_path: str | Path) -> Any:
    """Load an image as RGB without keeping the file handle open."""

    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError(
            "Pillow is required for image inference; install project dependencies first"
        ) from exc

    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(f"Image does not exist: {path}")
    try:
        with Image.open(path) as image:
            return image.convert("RGB")
    except OSError as exc:
        raise ValueError(f"Could not read image: {path}") from exc


def preprocess_image(image_path: str | Path, transform: Callable[[Any], Any]) -> Any:
    """Apply the transform supplied by the selected visual encoder."""

    return transform(load_rgb_image(image_path))
