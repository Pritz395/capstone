"""BUSI ultrasound image catalog loader.

Images are a different modality from WDBC tabular features — catalog only here.
CNN training belongs in a later week; this loader contributes VALID sample counts
with provenance for the multi-dataset corpus.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.datasets.base import DatasetSpec, LoadedDataset, assert_manifest, normalize_binary_label


def load(spec: DatasetSpec, root: Path) -> LoadedDataset:
    data_dir = root / spec.path
    parquet = data_dir / "raw" / "data" / "train-00000-of-00001.parquet"
    if not parquet.exists():
        folder_manifest = _from_folders(spec, data_dir)
        if folder_manifest is not None:
            return folder_manifest
        raise FileNotFoundError(
            f"BUSI data missing under {data_dir}. Run: python scripts/fetch_datasets.py --busi"
        )

    df = pd.read_parquet(parquet)
    required = {"class_label", "image_id"}
    if not required.issubset(df.columns):
        raise ValueError(f"BUSI parquet missing columns {required - set(df.columns)}")

    before = len(df)
    df = df.drop_duplicates(subset=["image_id"]).reset_index(drop=True)
    dropped_dupes = before - len(df)

    labels_raw = df["class_label"].astype(str).str.lower().str.strip()
    label_binary = []
    for lab in labels_raw:
        if lab == "normal":
            label_binary.append(pd.NA)
        else:
            mapped = normalize_binary_label(lab)
            if mapped is None:
                raise ValueError(f"BUSI: unknown class_label {lab!r}")
            label_binary.append(mapped)

    manifest = pd.DataFrame(
        {
            "sample_uid": [f"busi:{i}" for i in df["image_id"].astype(str)],
            "dataset_id": spec.dataset_id,
            "schema_id": spec.schema_id,
            "modality": spec.modality,
            "label_raw": labels_raw,
            "label_binary": pd.array(label_binary, dtype="Int64"),
            "source_path": str(parquet),
            "source_id": df["image_id"].astype(str),
        }
    )
    assert_manifest(manifest, spec.dataset_id)
    return LoadedDataset(
        spec=spec,
        manifest=manifest,
        features=None,
        labels=None,
        extras={
            "dropped_duplicates": int(dropped_dupes),
            "n_with_binary_label": int(manifest["label_binary"].notna().sum()),
        },
    )


def _from_folders(spec: DatasetSpec, data_dir: Path) -> LoadedDataset | None:
    rows = []
    for label in ("benign", "malignant", "normal"):
        folder = data_dir / label
        if not folder.exists():
            continue
        for img in sorted(folder.glob("*.png")) + sorted(folder.glob("*.jpg")):
            binary = None if label == "normal" else normalize_binary_label(label)
            rows.append(
                {
                    "sample_uid": f"busi:{img.stem}",
                    "dataset_id": spec.dataset_id,
                    "schema_id": spec.schema_id,
                    "modality": spec.modality,
                    "label_raw": label,
                    "label_binary": binary,
                    "source_path": str(img),
                    "source_id": img.stem,
                }
            )
    if not rows:
        return None
    manifest = pd.DataFrame(rows)
    manifest["label_binary"] = manifest["label_binary"].astype("Int64")
    assert_manifest(manifest, spec.dataset_id)
    return LoadedDataset(spec=spec, manifest=manifest, features=None, labels=None)
