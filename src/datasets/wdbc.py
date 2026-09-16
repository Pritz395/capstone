"""WDBC loader — preserves the existing 30-feature schema."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data import FEATURE_NAMES, load_wdbc
from src.datasets.base import DatasetSpec, LoadedDataset, assert_manifest, relative_source_path


def load(spec: DatasetSpec, root: Path) -> LoadedDataset:
    data_dir = root / spec.path
    csv_path = data_dir / "breast-cancer.csv"
    raw_path = data_dir / "wdbc.data"
    path = csv_path if csv_path.exists() else raw_path
    if not path.exists():
        raise FileNotFoundError(f"WDBC files not found under {data_dir}")

    # Keep original loader behavior (drops id, maps B/M → 0/1)
    df = load_wdbc(path)
    feature_cols = [c for c in FEATURE_NAMES if c in df.columns]
    if len(feature_cols) != 30:
        raise ValueError(f"WDBC expected 30 features, found {len(feature_cols)}")

    # Reload with id for stable provenance when possible
    if raw_path.exists():
        raw = pd.read_csv(raw_path, header=None, names=["id", "diagnosis", *FEATURE_NAMES])
        raw["diagnosis"] = raw["diagnosis"].map({"B": 0, "M": 1})
        ids = raw["id"].astype(str)
        y = raw["diagnosis"].astype(int)
        X = raw[feature_cols]
    else:
        ids = pd.Series([f"row{i}" for i in range(len(df))], index=df.index)
        y = df["diagnosis"].astype(int)
        X = df[feature_cols]

    # Deduplicate exact feature+label clones (keep first)
    keyed = pd.concat([X, y.rename("diagnosis")], axis=1)
    keep_idx = ~keyed.duplicated()
    X = X.loc[keep_idx].reset_index(drop=True)
    y = y.loc[keep_idx].reset_index(drop=True)
    ids = ids.loc[keep_idx].reset_index(drop=True)

    manifest = pd.DataFrame(
        {
            "sample_uid": [f"wdbc:{i}" for i in ids],
            "dataset_id": spec.dataset_id,
            "schema_id": spec.schema_id,
            "modality": spec.modality,
            "label_raw": y.map({0: "benign", 1: "malignant"}),
            "label_binary": y.astype("Int64"),
            "source_path": relative_source_path(path, root),
            "source_id": ids.astype(str),
        }
    )
    assert_manifest(manifest, spec.dataset_id)
    return LoadedDataset(spec=spec, manifest=manifest, features=X, labels=y)
