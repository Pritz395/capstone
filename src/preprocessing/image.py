"""Image preprocessing interface (Week 4 skeleton).

No CNN training here. This defines the future BUSI / BreakHis entry point so
Week 5+ can plug in without rewriting the tabular pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class ImageBatch:
    """Placeholder container for a future image training batch."""

    paths: list[Path]
    labels: list[int | None]
    meta: dict[str, Any]


class ImagePreprocessor:
    """Skeleton preprocessor for ultrasound / histopathology images.

    Future Week 5 responsibilities:
    - load PNG/JPG
    - resize / normalize
    - train-time augmentation
    - optional mask handling (BUSI)
    """

    def __init__(self, image_size: tuple[int, int] = (224, 224)):
        self.image_size = image_size

    def transform_path(self, path: Path) -> Path:
        """Identity stub — returns path unchanged until CNN pipeline lands."""
        if not path.exists():
            raise FileNotFoundError(path)
        return path

    def describe(self) -> dict[str, Any]:
        return {
            "status": "skeleton",
            "image_size": self.image_size,
            "implemented": False,
            "intended_datasets": ["busi", "breakhis"],
        }
