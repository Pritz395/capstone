"""Image model interface (Week 4 skeleton only — no training).

Future Week 5+ will implement CNN / ResNet-50 / EfficientNet behind this API
without rewriting tabular code.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class ImageEncoder(ABC):
    """Encodes an image into a latent vector or class logits (future)."""

    @abstractmethod
    def encode(self, image_batch: Any) -> Any:
        raise NotImplementedError

    @abstractmethod
    def predict_proba(self, image_batch: Any) -> Any:
        raise NotImplementedError

    def describe(self) -> dict[str, Any]:
        return {"role": "image_encoder", "implemented": False}


class ImageEncoderSkeleton(ImageEncoder):
    """Placeholder so imports/docs/tests can reference the future image branch."""

    def __init__(self, backbone: str = "resnet50_planned"):
        self.backbone = backbone

    def encode(self, image_batch: Any) -> Any:
        raise NotImplementedError(
            "Image encoder not trained yet (Week 5+). "
            f"Planned backbone: {self.backbone}"
        )

    def predict_proba(self, image_batch: Any) -> Any:
        raise NotImplementedError(
            "Image classifier not trained yet (Week 5+). "
            f"Planned backbone: {self.backbone}"
        )

    def describe(self) -> dict[str, Any]:
        return {
            "status": "skeleton",
            "backbone": self.backbone,
            "implemented": False,
            "planned_candidates": ["cnn_baseline", "resnet50", "efficientnet"],
            "xai_planned": ["gradcam", "gradcam++"],
        }
