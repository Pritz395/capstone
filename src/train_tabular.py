"""Train tabular baselines for one schema at a time (WDBC or WBCD).

Does NOT train on the full 2,024 corpus as one dataframe.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import roc_auc_score, roc_curve

from src.data import load_wdbc
from src.datasets.base import DatasetSpec
from src.datasets.registry import ROOT
from src.datasets.wbcd_original import load as load_wbcd
from src.models.tabular import (
    build_tabular_models,
    evaluate_tabular_model,
    select_best_model_name,
)
from src.preprocessing.tabular import prepare_tabular_splits

ARTIFACTS_ROOT = ROOT / "artifacts"
MODELS_ROOT = ROOT / "models"


def _plot_cms(results: dict, out_path: Path, title_prefix: str) -> None:
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
    fig.suptitle(f"{title_prefix} — confusion matrices", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_rocs(models: dict, X_test, y_test, out_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, model in models.items():
        if not hasattr(model, "predict_proba"):
            continue
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _load_xy(dataset: str):
    if dataset == "wdbc":
        df = load_wdbc()
        feature_cols = [c for c in df.columns if c != "diagnosis"]
        return df[feature_cols], df["diagnosis"].astype(int), "wdbc"
    if dataset == "wbcd":
        spec = DatasetSpec(
            dataset_id="wbcd_original",
            modality="tabular_cytology_ordinal",
            schema_id="wbcd_v1_9features",
            path="data/wbcd",
            task="binary_benign_malignant",
        )
        loaded = load_wbcd(spec, ROOT)
        assert loaded.features is not None and loaded.labels is not None
        return loaded.features, loaded.labels.astype(int), "wbcd"
    raise ValueError(f"Unsupported dataset for tabular training: {dataset}")


def train_tabular_dataset(dataset: str = "wdbc", random_state: int = 42) -> pd.DataFrame:
    dataset = dataset.lower().strip()
    X, y, key = _load_xy(dataset)

    art_dir = ARTIFACTS_ROOT / key
    model_dir = MODELS_ROOT / "tabular" / key
    art_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    split = prepare_tabular_splits(X, y, random_state=random_state)
    models = build_tabular_models(random_state=random_state)
    results: dict[str, dict] = {}
    trained: dict = {}

    print(f"Dataset: {key} | train={len(split.X_train)} test={len(split.X_test)} feats={len(split.feature_cols)}")
    print("-" * 72)

    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(split.X_train, split.y_train)
        metrics = evaluate_tabular_model(model, split.X_test, split.y_test)
        results[name] = metrics
        trained[name] = model
        joblib.dump(model, model_dir / f"{name.lower().replace(' ', '_')}.joblib")
        print(
            f"  Acc={metrics['accuracy']:.4f}  Prec={metrics['precision']:.4f}  "
            f"Rec={metrics['recall']:.4f}  Spec={metrics['specificity']:.4f}  "
            f"F1={metrics['f1']:.4f}  AUC={metrics['roc_auc']:.4f}"
        )

    leaderboard = (
        pd.DataFrame(
            [
                {
                    "model": name,
                    "accuracy": m["accuracy"],
                    "precision": m["precision"],
                    "recall": m["recall"],
                    "specificity": m["specificity"],
                    "f1": m["f1"],
                    "roc_auc": m["roc_auc"],
                }
                for name, m in results.items()
            ]
        )
        .sort_values(["accuracy", "roc_auc", "f1"], ascending=False)
        .reset_index(drop=True)
    )
    best_name = select_best_model_name(results, leaderboard["model"].tolist())
    best_model = trained[best_name]

    joblib.dump(split.scaler, model_dir / "scaler.joblib")
    joblib.dump(split.feature_cols, model_dir / "feature_cols.joblib")
    joblib.dump(best_model, model_dir / "best_model.joblib")
    (model_dir / "best_model_name.txt").write_text(best_name)

    # Keep legacy demo paths for WDBC Streamlit/Flask compatibility
    if key == "wdbc":
        MODELS_ROOT.mkdir(parents=True, exist_ok=True)
        joblib.dump(best_model, MODELS_ROOT / "best_model.joblib")
        joblib.dump(split.scaler, MODELS_ROOT / "scaler.joblib")
        joblib.dump(split.feature_cols, MODELS_ROOT / "feature_cols.joblib")
        (MODELS_ROOT / "best_model_name.txt").write_text(best_name)
        # Also mirror leaderboard at artifacts root for older docs/UI
        leaderboard.to_csv(ARTIFACTS_ROOT / "leaderboard.csv", index=False)
        with open(ARTIFACTS_ROOT / "metrics.json", "w") as f:
            json.dump(results, f, indent=2)

    leaderboard.to_csv(art_dir / "leaderboard.csv", index=False)
    with open(art_dir / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    _plot_cms(results, art_dir / "confusion_matrices.png", key.upper())
    _plot_rocs(
        trained,
        split.X_test,
        split.y_test,
        art_dir / "roc_curves.png",
        f"ROC — {key.upper()} tabular baselines",
    )

    meta = {
        "dataset": key,
        "n_samples": int(len(X)),
        "n_features": int(len(split.feature_cols)),
        "feature_cols": split.feature_cols,
        "train_size": int(len(split.X_train)),
        "test_size": int(len(split.X_test)),
        "best_model": best_name,
        "note": "Trained on this schema only — not on the full multi-dataset corpus.",
    }
    with open(art_dir / "run_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print("-" * 72)
    print(leaderboard.to_string(index=False))
    print(f"\nBest model ({key}): {best_name}")
    print(f"Artifacts → {art_dir}")
    print(f"Models → {model_dir}")
    return leaderboard


def main():
    parser = argparse.ArgumentParser(description="Train tabular baselines (one schema)")
    parser.add_argument(
        "--dataset",
        choices=["wdbc", "wbcd", "all"],
        default="all",
        help="Which tabular schema to train",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    targets = ["wdbc", "wbcd"] if args.dataset == "all" else [args.dataset]
    for ds in targets:
        train_tabular_dataset(ds, random_state=args.seed)


if __name__ == "__main__":
    main()
