"""Tabular preprocessing shared by WDBC / WBCD (and future tabular schemas).

Each schema stays independent — this module never merges incompatible columns.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


@dataclass
class TabularSplit:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    scaler: StandardScaler
    feature_cols: list[str]


def prepare_tabular_splits(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
) -> TabularSplit:
    """Stratified split + StandardScaler fit on train only (no leakage)."""
    if len(X) != len(y):
        raise ValueError("X and y length mismatch")
    if y.isna().any():
        raise ValueError("y contains missing labels — refuse to train")

    feature_cols = list(X.columns)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y.astype(int),
        test_size=test_size,
        random_state=random_state,
        stratify=y.astype(int),
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
    return TabularSplit(
        X_train=X_train_s,
        X_test=X_test_s,
        y_train=y_train,
        y_test=y_test,
        scaler=scaler,
        feature_cols=feature_cols,
    )
