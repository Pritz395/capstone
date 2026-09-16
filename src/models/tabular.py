"""Tabular model zoo + evaluation for WDBC / WBCD baselines."""

from __future__ import annotations

from typing import Any

import numpy as np
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


def build_tabular_models(random_state: int = 42) -> dict[str, Any]:
    return {
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=random_state),
        "Decision Tree": DecisionTreeClassifier(random_state=random_state),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, random_state=random_state, n_jobs=-1
        ),
        "SVM": SVC(kernel="rbf", probability=True, random_state=random_state),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "XGBoost": XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=random_state,
            n_jobs=-1,
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            random_state=random_state,
            verbose=-1,
            n_jobs=-1,
        ),
        "CatBoost": CatBoostClassifier(
            iterations=300,
            learning_rate=0.05,
            depth=6,
            verbose=False,
            random_state=random_state,
        ),
        "MLP": MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            max_iter=1000,
            random_state=random_state,
        ),
    }


def _specificity(cm: np.ndarray) -> float:
    # [[TN, FP], [FN, TP]]
    tn, fp = float(cm[0, 0]), float(cm[0, 1])
    denom = tn + fp
    return float(tn / denom) if denom else 0.0


def evaluate_tabular_model(model, X_test, y_test) -> dict[str, Any]:
    y_pred = model.predict(X_test)
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        y_prob = y_pred.astype(float)

    cm = confusion_matrix(y_test, y_pred)
    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "specificity": _specificity(cm),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
        "confusion_matrix": cm.tolist(),
        "classification_report": classification_report(
            y_test, y_pred, target_names=["Benign", "Malignant"], output_dict=True
        ),
    }


def select_best_model_name(results: dict[str, dict], leaderboard_order: list[str]) -> str:
    """Prefer tree ensembles on accuracy ties (better for later SHAP)."""
    tree_preference = [
        "Random Forest",
        "CatBoost",
        "LightGBM",
        "XGBoost",
        "Decision Tree",
    ]
    top_acc = max(m["accuracy"] for m in results.values())
    tied = [n for n, m in results.items() if m["accuracy"] >= top_acc - 1e-9]
    for preferred in tree_preference:
        if preferred in tied:
            return preferred
    return leaderboard_order[0]
