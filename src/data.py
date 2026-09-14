"""Load and prepare the Wisconsin Diagnostic Breast Cancer (WDBC) dataset."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

FEATURE_NAMES = [
    "radius_mean",
    "texture_mean",
    "perimeter_mean",
    "area_mean",
    "smoothness_mean",
    "compactness_mean",
    "concavity_mean",
    "concave_points_mean",
    "symmetry_mean",
    "fractal_dimension_mean",
    "radius_se",
    "texture_se",
    "perimeter_se",
    "area_se",
    "smoothness_se",
    "compactness_se",
    "concavity_se",
    "concave_points_se",
    "symmetry_se",
    "fractal_dimension_se",
    "radius_worst",
    "texture_worst",
    "perimeter_worst",
    "area_worst",
    "smoothness_worst",
    "compactness_worst",
    "concavity_worst",
    "concave_points_worst",
    "symmetry_worst",
    "fractal_dimension_worst",
]

FEATURE_MEANINGS = {
    "radius": "Mean of distances from center to points on the nucleus perimeter",
    "texture": "Standard deviation of gray-scale values in the nucleus",
    "perimeter": "Perimeter length of the nucleus contour",
    "area": "Area of the nucleus",
    "smoothness": "Local variation in radius lengths",
    "compactness": "perimeter^2 / area - 1.0",
    "concavity": "Severity of concave portions of the contour",
    "concave_points": "Number of concave portions of the contour",
    "symmetry": "Symmetry of the nucleus shape",
    "fractal_dimension": "Coastline approximation - 1 (complexity of contour)",
}

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = ROOT / "data" / "wdbc" / "breast-cancer.csv"
DEFAULT_RAW = ROOT / "data" / "wdbc" / "wdbc.data"


def load_wdbc(csv_path: Path | None = None) -> pd.DataFrame:
    """Load WDBC as a clean dataframe with diagnosis as 0/1 (B=0, M=1)."""
    if csv_path is None:
        if DEFAULT_CSV.exists():
            csv_path = DEFAULT_CSV
        else:
            csv_path = DEFAULT_RAW

    if csv_path.name.endswith(".data") or csv_path == DEFAULT_RAW:
        columns = ["id", "diagnosis", *FEATURE_NAMES]
        df = pd.read_csv(csv_path, header=None, names=columns)
    else:
        df = pd.read_csv(csv_path)
        # Normalize common Kaggle / CSV header variants
        rename = {c: c.strip().replace(" ", "_") for c in df.columns}
        df = df.rename(columns=rename)
        if "Unnamed:_32" in df.columns:
            df = df.drop(columns=["Unnamed:_32"])
        if "Unnamed: 32" in df.columns:
            df = df.drop(columns=["Unnamed: 32"])

    if "id" in df.columns:
        df = df.drop(columns=["id"])

    df["diagnosis"] = df["diagnosis"].map({"B": 0, "M": 1, "benign": 0, "malignant": 1})
    if df["diagnosis"].isna().any():
        raise ValueError("Could not map diagnosis labels to 0/1")

    feature_cols = [c for c in FEATURE_NAMES if c in df.columns]
    if len(feature_cols) != 30:
        # Fall back to all non-label columns
        feature_cols = [c for c in df.columns if c != "diagnosis"]

    df = df[["diagnosis", *feature_cols]].dropna()
    return df


def prepare_splits(
    df: pd.DataFrame | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """Return scaled train/test splits and the fitted scaler."""
    if df is None:
        df = load_wdbc()

    feature_cols = [c for c in df.columns if c != "diagnosis"]
    X = df[feature_cols]
    y = df["diagnosis"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    scaler = StandardScaler()
    X_train_s = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=feature_cols,
        index=X_train.index,
    )
    X_test_s = pd.DataFrame(
        scaler.transform(X_test),
        columns=feature_cols,
        index=X_test.index,
    )
    return X_train_s, X_test_s, y_train, y_test, scaler, feature_cols
