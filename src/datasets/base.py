"""Shared types and validation helpers for dataset loaders."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

REQUIRED_MANIFEST_COLUMNS = [
    "sample_uid",
    "dataset_id",
    "schema_id",
    "modality",
    "label_raw",
    "label_binary",
    "source_path",
]


@dataclass(frozen=True)
class DatasetSpec:
    dataset_id: str
    modality: str
    schema_id: str
    path: str
    task: str
    enabled: bool = True
    notes: str = ""


@dataclass
class LoadedDataset:
    """A loaded dataset with provenance-ready rows and optional model matrix."""

    spec: DatasetSpec
    manifest: pd.DataFrame
    # Feature matrix used for THIS schema only (may be None for image catalogs)
    features: pd.DataFrame | None = None
    labels: pd.Series | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    def n_samples(self) -> int:
        return int(len(self.manifest))


def assert_manifest(df: pd.DataFrame, dataset_id: str) -> None:
    missing = [c for c in REQUIRED_MANIFEST_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{dataset_id}: manifest missing columns {missing}")
    if df["sample_uid"].duplicated().any():
        raise ValueError(f"{dataset_id}: duplicate sample_uid values")
    if df["dataset_id"].ne(dataset_id).any():
        raise ValueError(f"{dataset_id}: dataset_id column mismatch")


def normalize_binary_label(value) -> int | None:
    """Map common benign/malignant encodings to 0/1. Returns None if unknown."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        v = int(value)
        if v in (0, 1):
            return v
        if v == 2:  # WBCD benign
            return 0
        if v == 4:  # WBCD malignant
            return 1
    s = str(value).strip().lower()
    mapping = {
        "b": 0,
        "benign": 0,
        "0": 0,
        "2": 0,
        "m": 1,
        "malignant": 1,
        "1": 1,
        "4": 1,
    }
    return mapping.get(s)
