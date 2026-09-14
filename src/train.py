"""Train and compare ML/DL models for WDBC breast cancer classification."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
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
    roc_curve,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from src.data import prepare_splits

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
ARTIFACTS_DIR = ROOT / "artifacts"


def build_models(random_state: int = 42) -> dict:
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


def evaluate_model(model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        y_prob = y_pred.astype(float)

    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(
            y_test, y_pred, target_names=["Benign", "Malignant"], output_dict=True
        ),
    }


def plot_confusion_matrices(results: dict, out_path: Path) -> None:
    n = len(results)
    cols = 3
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.5 * rows))
    axes = np.array(axes).reshape(-1)
    for ax, (name, metrics) in zip(axes, results.items()):
        cm = np.array(metrics["confusion_matrix"])
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Benign", "Malignant"],
            yticklabels=["Benign", "Malignant"],
            ax=ax,
            cbar=False,
        )
        ax.set_title(f"{name}\nAcc={metrics['accuracy']:.3f}")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
    for ax in axes[n:]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_roc_curves(models: dict, X_test, y_test, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, model in models.items():
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
        else:
            continue
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — WDBC Breast Cancer Classification")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def train_all(random_state: int = 42) -> pd.DataFrame:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    X_train, X_test, y_train, y_test, scaler, feature_cols = prepare_splits(
        random_state=random_state
    )

    models = build_models(random_state=random_state)
    results = {}
    trained = {}

    print(f"Train size: {len(X_train)} | Test size: {len(X_test)}")
    print(f"Features: {len(feature_cols)}")
    print("-" * 72)

    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test)
        results[name] = metrics
        trained[name] = model
        safe = name.lower().replace(" ", "_")
        joblib.dump(model, MODELS_DIR / f"{safe}.joblib")
        print(
            f"  Acc={metrics['accuracy']:.4f}  "
            f"Prec={metrics['precision']:.4f}  "
            f"Rec={metrics['recall']:.4f}  "
            f"F1={metrics['f1']:.4f}  "
            f"AUC={metrics['roc_auc']:.4f}"
        )

    joblib.dump(scaler, MODELS_DIR / "scaler.joblib")
    joblib.dump(feature_cols, MODELS_DIR / "feature_cols.joblib")

    leaderboard = (
        pd.DataFrame(
            [
                {
                    "model": name,
                    "accuracy": m["accuracy"],
                    "precision": m["precision"],
                    "recall": m["recall"],
                    "f1": m["f1"],
                    "roc_auc": m["roc_auc"],
                }
                for name, m in results.items()
            ]
        )
        .sort_values(["accuracy", "roc_auc", "f1"], ascending=False)
        .reset_index(drop=True)
    )

    # Prefer tree ensembles when accuracy is effectively tied — SHAP TreeExplainer
    # is fast and matches the XAI literature for this project.
    tree_preference = [
        "Random Forest",
        "CatBoost",
        "LightGBM",
        "XGBoost",
        "Decision Tree",
    ]
    top_acc = float(leaderboard.iloc[0]["accuracy"])
    tied_names = [
        name
        for name, m in results.items()
        if m["accuracy"] >= top_acc - 1e-9
    ]
    best_name = leaderboard.iloc[0]["model"]
    for preferred in tree_preference:
        if preferred in tied_names:
            best_name = preferred
            break
    best_model = trained[best_name]
    print(f"Tied at ~{top_acc:.4f}: {tied_names} → selected {best_name}")
    joblib.dump(best_model, MODELS_DIR / "best_model.joblib")
    with open(MODELS_DIR / "best_model_name.txt", "w") as f:
        f.write(best_name)

    leaderboard.to_csv(ARTIFACTS_DIR / "leaderboard.csv", index=False)
    with open(ARTIFACTS_DIR / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    plot_confusion_matrices(results, ARTIFACTS_DIR / "confusion_matrices.png")
    plot_roc_curves(trained, X_test, y_test, ARTIFACTS_DIR / "roc_curves.png")

    print("-" * 72)
    print("Leaderboard:")
    print(leaderboard.to_string(index=False))
    print(f"\nBest model: {best_name}")
    print(f"Saved models → {MODELS_DIR}")
    print(f"Saved artifacts → {ARTIFACTS_DIR}")
    return leaderboard


if __name__ == "__main__":
    train_all()
