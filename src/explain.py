"""Generate SHAP and LIME explanations for the best trained model."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from lime.lime_tabular import LimeTabularExplainer

from src.data import prepare_splits

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
ARTIFACTS_DIR = ROOT / "artifacts"


def load_artifacts():
    best_path = MODELS_DIR / "best_model.joblib"
    if not best_path.exists():
        raise FileNotFoundError("Train models first: python -m src.train")
    model = joblib.load(best_path)
    scaler = joblib.load(MODELS_DIR / "scaler.joblib")
    feature_cols = joblib.load(MODELS_DIR / "feature_cols.joblib")
    best_name = (MODELS_DIR / "best_model_name.txt").read_text().strip()
    return model, scaler, feature_cols, best_name


def _predict_proba_fn(model):
    def fn(X):
        X = np.asarray(X)
        return model.predict_proba(X)

    return fn


def build_shap_explanations(max_samples: int = 100) -> dict:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    model, scaler, feature_cols, best_name = load_artifacts()
    X_train, X_test, y_train, y_test, _, _ = prepare_splits()

    # Background for KernelExplainer / general use
    background = shap.sample(X_train, min(50, len(X_train)), random_state=42)
    explain_X = X_test.iloc[: min(max_samples, len(X_test))]

    tree_models = {
        "Random Forest",
        "Decision Tree",
        "XGBoost",
        "LightGBM",
        "CatBoost",
    }

    if best_name in tree_models:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(explain_X)
        if isinstance(shap_values, list):
            # binary classifiers sometimes return [neg, pos]
            values = shap_values[1] if len(shap_values) == 2 else shap_values[0]
        else:
            values = shap_values
            if getattr(values, "ndim", 1) == 3:
                values = values[:, :, 1]
    else:
        explainer = shap.Explainer(model.predict_proba, background)
        explanation = explainer(explain_X)
        values = explanation.values
        if values.ndim == 3:
            values = values[:, :, 1]

    values = np.asarray(values)
    mean_abs = np.abs(values).mean(axis=0)
    importance = (
        pd.DataFrame({"feature": feature_cols, "mean_abs_shap": mean_abs})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    importance.to_csv(ARTIFACTS_DIR / "shap_feature_importance.csv", index=False)

    # Summary bar plot
    plt.figure(figsize=(8, 7))
    top = importance.head(15)
    plt.barh(top["feature"][::-1], top["mean_abs_shap"][::-1], color="#2a6f97")
    plt.xlabel("Mean |SHAP value|")
    plt.title(f"SHAP Feature Importance — {best_name}")
    plt.tight_layout()
    plt.savefig(ARTIFACTS_DIR / "shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Beeswarm if possible
    try:
        plt.figure(figsize=(9, 7))
        shap.summary_plot(values, explain_X, show=False, max_display=15)
        plt.tight_layout()
        plt.savefig(ARTIFACTS_DIR / "shap_beeswarm.png", dpi=150, bbox_inches="tight")
        plt.close()
    except Exception as exc:  # noqa: BLE001
        print(f"Beeswarm plot skipped: {exc}")

    # Local explanation for first malignant test case if present
    malignant_idx = y_test[y_test == 1].index
    if len(malignant_idx):
        sample_idx = malignant_idx[0]
    else:
        sample_idx = y_test.index[0]
    sample = X_test.loc[[sample_idx]]
    local_values = np.asarray(
        explainer.shap_values(sample)
        if best_name in tree_models
        else explainer(sample).values
    )
    if isinstance(local_values, list):
        local_values = local_values[1]
    local_values = np.asarray(local_values).reshape(-1)
    if local_values.ndim > 1:
        local_values = local_values[:, 1] if local_values.shape[-1] == 2 else local_values.ravel()

    local = (
        pd.DataFrame(
            {
                "feature": feature_cols,
                "value": sample.iloc[0].values,
                "shap": local_values[: len(feature_cols)],
            }
        )
        .assign(abs_shap=lambda d: d["shap"].abs())
        .sort_values("abs_shap", ascending=False)
        .head(10)
        .drop(columns=["abs_shap"])
    )
    local.to_csv(ARTIFACTS_DIR / "shap_local_example.csv", index=False)

    # LIME for the same sample
    lime_explainer = LimeTabularExplainer(
        training_data=X_train.values,
        feature_names=feature_cols,
        class_names=["Benign", "Malignant"],
        mode="classification",
        discretize_continuous=True,
        random_state=42,
    )
    lime_exp = lime_explainer.explain_instance(
        sample.iloc[0].values,
        _predict_proba_fn(model),
        num_features=10,
    )
    lime_exp.save_to_file(str(ARTIFACTS_DIR / "lime_local_example.html"))
    lime_pairs = lime_exp.as_list(label=1)
    with open(ARTIFACTS_DIR / "lime_local_example.json", "w") as f:
        json.dump([{"feature": a, "weight": b} for a, b in lime_pairs], f, indent=2)

    summary = {
        "model": best_name,
        "top_global_features": importance.head(10).to_dict(orient="records"),
        "local_sample_index": int(sample_idx) if isinstance(sample_idx, (int, np.integer)) else str(sample_idx),
        "local_true_label": int(y_test.loc[sample_idx]),
        "local_prediction": int(model.predict(sample)[0]),
        "local_probability_malignant": float(model.predict_proba(sample)[0][1]),
    }
    with open(ARTIFACTS_DIR / "explanation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"SHAP/LIME artifacts saved → {ARTIFACTS_DIR}")
    print(f"Best model explained: {best_name}")
    print(importance.head(10).to_string(index=False))
    return summary


if __name__ == "__main__":
    build_shap_explanations()
