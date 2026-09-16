"""BreakHis histopathology image catalog loader (optional, path to 20K).

Official source: UFPR BreakHis (~7,909 images). Enable in configs/datasets.yaml
after placing the archive under data/breakhis/.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.datasets.base import DatasetSpec, LoadedDataset, assert_manifest, normalize_binary_label


def load(spec: DatasetSpec, root: Path) -> LoadedDataset:
    data_dir = root / spec.path
    if not data_dir.exists():
        raise FileNotFoundError(
            f"BreakHis directory missing: {data_dir}. "
            "Download from http://www.inf.ufpr.br/vri/databases/BreaKHis_v1.tar.gz "
            "and extract under data/breakhis/"
        )

    images = list(data_dir.rglob("*.png")) + list(data_dir.rglob("*.jpg"))
    if not images:
        raise FileNotFoundError(f"No images found under {data_dir}")

    rows = []
    for img in sorted(images):
        parts_lower = str(img).lower()
        if "benign" in parts_lower:
            raw = "benign"
        elif "malignant" in parts_lower:
            raw = "malignant"
        else:
            # Skip unmarked files rather than inventing labels
            continue
        rows.append(
            {
                "sample_uid": f"breakhis:{img.relative_to(data_dir).as_posix()}",
                "dataset_id": spec.dataset_id,
                "schema_id": spec.schema_id,
                "modality": spec.modality,
                "label_raw": raw,
                "label_binary": normalize_binary_label(raw),
                "source_path": str(img),
                "source_id": img.stem,
            }
        )

    if not rows:
        raise ValueError("BreakHis: found images but could not infer benign/malignant from paths")

    manifest = pd.DataFrame(rows).drop_duplicates(subset=["sample_uid"]).reset_index(drop=True)
    manifest["label_binary"] = manifest["label_binary"].astype("Int64")
    assert_manifest(manifest, spec.dataset_id)
    return LoadedDataset(spec=spec, manifest=manifest, features=None, labels=None)
