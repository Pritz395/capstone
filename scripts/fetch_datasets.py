#!/usr/bin/env python3
"""Fetch open datasets needed for the multi-source corpus (no fabricated samples)."""

from __future__ import annotations

import argparse
from pathlib import Path
from urllib.request import urlretrieve

ROOT = Path(__file__).resolve().parents[1]

WBCD_DATA = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "breast-cancer-wisconsin/breast-cancer-wisconsin.data"
)
WBCD_NAMES = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "breast-cancer-wisconsin/breast-cancer-wisconsin.names"
)


def fetch_wbcd() -> Path:
    out = ROOT / "data" / "wbcd"
    out.mkdir(parents=True, exist_ok=True)
    data_path = out / "breast-cancer-wisconsin.data"
    names_path = out / "breast-cancer-wisconsin.names"
    if not data_path.exists():
        print(f"Downloading WBCD → {data_path}")
        urlretrieve(WBCD_DATA, data_path)
    if not names_path.exists():
        urlretrieve(WBCD_NAMES, names_path)
    print(f"WBCD ready: {data_path} ({sum(1 for _ in open(data_path))} lines)")
    return data_path


def fetch_busi() -> Path:
    """Download BUSI mirror from Hugging Face (MedOtter/BUSI, 780 images)."""
    from huggingface_hub import snapshot_download

    out = ROOT / "data" / "busi" / "raw"
    out.mkdir(parents=True, exist_ok=True)
    print("Downloading BUSI from Hugging Face (MedOtter/BUSI)…")
    path = snapshot_download(repo_id="MedOtter/BUSI", repo_type="dataset", local_dir=str(out))
    parquet = out / "data" / "train-00000-of-00001.parquet"
    if not parquet.exists():
        raise FileNotFoundError(f"Expected parquet missing after download: {parquet}")
    print(f"BUSI ready: {parquet}")
    return parquet


def main():
    parser = argparse.ArgumentParser(description="Fetch corpus datasets")
    parser.add_argument("--wbcd", action="store_true", help="Fetch UCI Wisconsin Original")
    parser.add_argument("--busi", action="store_true", help="Fetch BUSI ultrasound (HF)")
    parser.add_argument("--all", action="store_true", help="Fetch WBCD + BUSI")
    args = parser.parse_args()
    if not any([args.wbcd, args.busi, args.all]):
        parser.print_help()
        return
    if args.all or args.wbcd:
        fetch_wbcd()
    if args.all or args.busi:
        fetch_busi()


if __name__ == "__main__":
    main()
