"""Dataset registry: load configured datasets and build a provenance corpus.

CRITICAL: This module never concatenates incompatible feature matrices.
Training matrices are returned per schema_id only.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import pandas as pd
import yaml

from src.datasets.base import REQUIRED_MANIFEST_COLUMNS, DatasetSpec, LoadedDataset
from src.datasets import busi, breakhis, wbcd_original, wdbc

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs" / "datasets.yaml"

LOADERS: dict[str, Callable[[DatasetSpec, Path], LoadedDataset]] = {
    "wdbc": wdbc.load,
    "wbcd_original": wbcd_original.load,
    "busi": busi.load,
    "breakhis": breakhis.load,
}


def load_config(config_path: Path | None = None) -> dict:
    path = config_path or DEFAULT_CONFIG
    with open(path) as f:
        return yaml.safe_load(f)


def iter_specs(config: dict | None = None) -> list[DatasetSpec]:
    cfg = config or load_config()
    specs = []
    for dataset_id, meta in cfg["datasets"].items():
        specs.append(
            DatasetSpec(
                dataset_id=dataset_id,
                modality=meta["modality"],
                schema_id=meta["schema_id"],
                path=meta["path"],
                task=meta["task"],
                enabled=bool(meta.get("enabled", True)),
                notes=meta.get("notes", ""),
            )
        )
    return specs


def load_enabled(config_path: Path | None = None, root: Path | None = None) -> dict[str, LoadedDataset]:
    cfg = load_config(config_path)
    root = root or ROOT
    loaded: dict[str, LoadedDataset] = {}
    for spec in iter_specs(cfg):
        if not spec.enabled:
            continue
        loader = LOADERS.get(spec.dataset_id)
        if loader is None:
            raise KeyError(f"No loader registered for dataset_id={spec.dataset_id}")
        loaded[spec.dataset_id] = loader(spec, root)
    if not loaded:
        raise RuntimeError("No datasets enabled in config")
    return loaded


def build_corpus_manifest(loaded: dict[str, LoadedDataset]) -> pd.DataFrame:
    frames = [ds.manifest for ds in loaded.values()]
    corpus = pd.concat(frames, ignore_index=True)
    missing = [c for c in REQUIRED_MANIFEST_COLUMNS if c not in corpus.columns]
    if missing:
        raise ValueError(f"Corpus manifest missing columns {missing}")
    if corpus["sample_uid"].duplicated().any():
        dupes = corpus.loc[corpus["sample_uid"].duplicated(keep=False), "sample_uid"].head(10).tolist()
        raise ValueError(f"Corpus has colliding sample_uid values: {dupes}")
    return corpus


def schema_feature_matrix(
    loaded: dict[str, LoadedDataset], schema_id: str
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Return X, y, manifest rows for a SINGLE schema only."""
    parts_x = []
    parts_y = []
    parts_m = []
    for ds in loaded.values():
        if ds.spec.schema_id != schema_id:
            continue
        if ds.features is None or ds.labels is None:
            raise ValueError(
                f"Schema {schema_id} includes {ds.spec.dataset_id} without a tabular feature matrix"
            )
        parts_x.append(ds.features.reset_index(drop=True))
        parts_y.append(ds.labels.reset_index(drop=True))
        parts_m.append(ds.manifest.reset_index(drop=True))
    if not parts_x:
        raise KeyError(f"No tabular datasets found for schema_id={schema_id}")

    # Guard: all feature column sets must match exactly
    cols0 = list(parts_x[0].columns)
    for x in parts_x[1:]:
        if list(x.columns) != cols0:
            raise ValueError(
                f"Refusing to merge schema {schema_id}: feature columns differ "
                f"{cols0} vs {list(x.columns)}"
            )

    X = pd.concat(parts_x, ignore_index=True)
    y = pd.concat(parts_y, ignore_index=True).astype(int)
    manifest = pd.concat(parts_m, ignore_index=True)
    return X, y, manifest


def corpus_summary(corpus: pd.DataFrame, loaded: dict[str, LoadedDataset]) -> pd.DataFrame:
    rows = []
    for dataset_id, ds in loaded.items():
        sub = corpus[corpus["dataset_id"] == dataset_id]
        rows.append(
            {
                "dataset_id": dataset_id,
                "schema_id": ds.spec.schema_id,
                "modality": ds.spec.modality,
                "n_samples": len(sub),
                "n_binary_labeled": int(sub["label_binary"].notna().sum()),
                "n_benign": int((sub["label_binary"] == 0).sum()),
                "n_malignant": int((sub["label_binary"] == 1).sum()),
                "has_feature_matrix": ds.features is not None,
            }
        )
    rows.append(
        {
            "dataset_id": "ALL",
            "schema_id": "—",
            "modality": "mixed",
            "n_samples": len(corpus),
            "n_binary_labeled": int(corpus["label_binary"].notna().sum()),
            "n_benign": int((corpus["label_binary"] == 0).sum()),
            "n_malignant": int((corpus["label_binary"] == 1).sum()),
            "has_feature_matrix": False,
        }
    )
    return pd.DataFrame(rows)
