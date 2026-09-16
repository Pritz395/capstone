#!/usr/bin/env python3
"""Validate corpus integrity and schema isolation rules."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.datasets.registry import (
    build_corpus_manifest,
    load_enabled,
    schema_feature_matrix,
)

# ROOT already defined above for sys.path


def main() -> int:
    loaded = load_enabled()
    corpus = build_corpus_manifest(loaded)
    errors = []

    if corpus["sample_uid"].isna().any():
        errors.append("null sample_uid")
    if corpus["sample_uid"].duplicated().any():
        errors.append("duplicate sample_uid")

    # Each dataset must have a single schema_id
    for dataset_id, ds in loaded.items():
        schemas = corpus.loc[corpus["dataset_id"] == dataset_id, "schema_id"].unique()
        if len(schemas) != 1:
            errors.append(f"{dataset_id} has multiple schema_ids: {schemas}")

    # Attempting to merge WDBC + WBCD features must fail loudly
    try:
        # Force a bad merge check by comparing columns
        if "wdbc" in loaded and "wbcd_original" in loaded:
            w_cols = list(loaded["wdbc"].features.columns)
            o_cols = list(loaded["wbcd_original"].features.columns)
            if w_cols == o_cols:
                errors.append("WDBC and WBCD unexpectedly share identical feature columns")
            # schema_feature_matrix for each schema should work independently
            schema_feature_matrix(loaded, loaded["wdbc"].spec.schema_id)
            schema_feature_matrix(loaded, loaded["wbcd_original"].spec.schema_id)
    except Exception as exc:  # noqa: BLE001
        errors.append(f"schema isolation check error: {exc}")

    # No fabricated labels
    if (corpus["label_raw"].astype(str).str.lower() == "nan").any():
        errors.append("fabricated/nan label_raw strings present")

    summary_path = ROOT / "artifacts" / "corpus" / "summary.csv"
    if not summary_path.exists():
        errors.append("missing artifacts/corpus/summary.csv — run scripts/build_corpus.py first")

    if errors:
        print("VALIDATION FAILED:")
        for e in errors:
            print(f"  - {e}")
        return 1

    print("VALIDATION PASSED")
    print(f"  corpus_n={len(corpus)}")
    print(f"  datasets={list(loaded.keys())}")
    print(f"  schemas={sorted({d.spec.schema_id for d in loaded.values()})}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
