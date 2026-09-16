"""Wisconsin Breast Cancer Original (WBCD) — 9 ordinal cytology features.

NOT schema-compatible with WDBC's 30 continuous morphometry features.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.datasets.base import DatasetSpec, LoadedDataset, assert_manifest, normalize_binary_label, relative_source_path

FEATURE_NAMES = [
    "clump_thickness",
    "uniformity_cell_size",
    "uniformity_cell_shape",
    "marginal_adhesion",
    "single_epithelial_cell_size",
    "bare_nuclei",
    "bland_chromatin",
    "normal_nucleoli",
    "mitoses",
]


def load(spec: DatasetSpec, root: Path) -> LoadedDataset:
    data_dir = root / spec.path
    path = data_dir / "breast-cancer-wisconsin.data"
    if not path.exists():
        raise FileNotFoundError(
            f"WBCD file missing: {path}. Run: python scripts/fetch_datasets.py --wbcd"
        )

    columns = ["sample_code_number", *FEATURE_NAMES, "class"]
    df = pd.read_csv(path, header=None, names=columns)

    # Bare nuclei uses '?' for missing — drop those rows (do not impute/fabricate)
    df["bare_nuclei"] = pd.to_numeric(df["bare_nuclei"], errors="coerce")
    before = len(df)
    df = df.dropna(subset=["bare_nuclei"]).copy()
    dropped_missing = before - len(df)

    for col in FEATURE_NAMES:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=FEATURE_NAMES).copy()

    df["label_binary"] = df["class"].map(normalize_binary_label)
    if df["label_binary"].isna().any():
        bad = df.loc[df["label_binary"].isna(), "class"].unique().tolist()
        raise ValueError(f"WBCD: unmapped class values {bad}")

    # Only remove exact full duplicates (same id + features + label).
    # Do NOT collapse different patients that happen to share cytology scores.
    before_dedup = len(df)
    df = df.drop_duplicates(
        subset=["sample_code_number", *FEATURE_NAMES, "label_binary"]
    ).reset_index(drop=True)
    dropped_dupes = before_dedup - len(df)

    X = df[FEATURE_NAMES].astype(float)
    y = df["label_binary"].astype(int)
    ids = df["sample_code_number"].astype(str)

    manifest = pd.DataFrame(
        {
            "sample_uid": [f"wbcd_original:{i}:{k}" for k, i in enumerate(ids)],
            "dataset_id": spec.dataset_id,
            "schema_id": spec.schema_id,
            "modality": spec.modality,
            "label_raw": y.map({0: "benign", 1: "malignant"}),
            "label_binary": y.astype("Int64"),
            "source_path": relative_source_path(path, root),
            "source_id": ids,
        }
    )
    assert_manifest(manifest, spec.dataset_id)
    return LoadedDataset(
        spec=spec,
        manifest=manifest,
        features=X,
        labels=y,
        extras={
            "dropped_missing": int(dropped_missing),
            "dropped_duplicates": int(dropped_dupes),
            "feature_names": FEATURE_NAMES,
        },
    )
