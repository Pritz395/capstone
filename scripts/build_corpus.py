#!/usr/bin/env python3
"""Build the multi-dataset corpus manifest and validate sample counts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.datasets.registry import (
    build_corpus_manifest,
    corpus_summary,
    load_config,
    load_enabled,
)

OUT_DIR = ROOT / "artifacts" / "corpus"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument(
        "--min-samples",
        type=int,
        default=None,
        help="Fail if corpus has fewer than this many VALID samples",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    min_samples = args.min_samples
    if min_samples is None:
        min_samples = int(cfg.get("project", {}).get("min_corpus_size", 0))

    loaded = load_enabled(args.config)
    corpus = build_corpus_manifest(loaded)
    summary = corpus_summary(corpus, loaded)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    corpus_path = OUT_DIR / "manifest.csv"
    summary_path = OUT_DIR / "summary.csv"
    meta_path = OUT_DIR / "build_meta.json"

    corpus.to_csv(corpus_path, index=False)
    summary.to_csv(summary_path, index=False)

    meta = {
        "n_samples": int(len(corpus)),
        "n_datasets": len(loaded),
        "datasets": list(loaded.keys()),
        "schemas": sorted({ds.spec.schema_id for ds in loaded.values()}),
        "min_samples_required": min_samples,
        "passes_min_samples": bool(len(corpus) >= min_samples),
        "note": (
            "Corpus counts VALID samples across modalities/schemas. "
            "Feature matrices are NEVER merged across incompatible schemas."
        ),
    }
    meta_path.write_text(json.dumps(meta, indent=2))

    print(summary.to_string(index=False))
    print("-" * 72)
    print(f"Total VALID corpus samples: {len(corpus)}")
    print(f"Wrote {corpus_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {meta_path}")

    if min_samples and len(corpus) < min_samples:
        raise SystemExit(
            f"FAIL: corpus has {len(corpus)} samples; mentor milestone requires >= {min_samples}. "
            "Enable/fetch additional datasets (see docs/DATASETS.md)."
        )
    print(f"PASS: corpus size {len(corpus)} >= {min_samples}")


if __name__ == "__main__":
    main()
