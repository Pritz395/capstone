"""Flask web UI for breast cancer prediction + SHAP-style explanations."""

from __future__ import annotations

import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request

from src.data import FEATURE_MEANINGS, FEATURE_NAMES

ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
ARTIFACTS_DIR = ROOT / "artifacts"

app = Flask(__name__)

_model = None
_scaler = None
_feature_cols = None
_best_name = None
_importance = None


def load_runtime():
    global _model, _scaler, _feature_cols, _best_name, _importance
    if _model is not None:
        return
    _model = joblib.load(MODELS_DIR / "best_model.joblib")
    _scaler = joblib.load(MODELS_DIR / "scaler.joblib")
    _feature_cols = joblib.load(MODELS_DIR / "feature_cols.joblib")
    name_path = MODELS_DIR / "best_model_name.txt"
    _best_name = name_path.read_text().strip() if name_path.exists() else "Random Forest"
    imp_path = ARTIFACTS_DIR / "shap_feature_importance.csv"
    if imp_path.exists():
        _importance = pd.read_csv(imp_path)
    else:
        _importance = None


def feature_base(name: str) -> str:
    for key in FEATURE_MEANINGS:
        if name.startswith(key):
            return key
    return name


@app.route("/health")
def health():
    try:
        load_runtime()
        return jsonify({"status": "ok", "model": _best_name}), 200
    except Exception as exc:  # noqa: BLE001
        return jsonify({"status": "error", "detail": str(exc)}), 500


@app.route("/")
def index():
    load_runtime()
    leaderboard = None
    lb_path = ARTIFACTS_DIR / "leaderboard.csv"
    if lb_path.exists():
        leaderboard = pd.read_csv(lb_path).to_dict(orient="records")
    return render_template(
        "index.html",
        feature_cols=_feature_cols,
        best_name=_best_name,
        leaderboard=leaderboard,
        importance=(
            _importance.head(10).to_dict(orient="records") if _importance is not None else []
        ),
        feature_meanings=FEATURE_MEANINGS,
    )


@app.route("/api/predict", methods=["POST"])
def predict():
    load_runtime()
    payload = request.get_json(force=True)
    values = []
    for col in _feature_cols:
        if col not in payload:
            return jsonify({"error": f"Missing feature: {col}"}), 400
        values.append(float(payload[col]))

    X = pd.DataFrame([values], columns=_feature_cols)
    Xs = pd.DataFrame(_scaler.transform(X), columns=_feature_cols)
    proba = float(_model.predict_proba(Xs)[0][1])
    pred = int(proba >= 0.5)
    label = "Malignant" if pred == 1 else "Benign"

    contributions = []
    try:
        import shap

        tree_names = {"Random Forest", "Decision Tree", "XGBoost", "LightGBM", "CatBoost"}
        if _best_name in tree_names:
            explainer = shap.TreeExplainer(_model)
            sv = explainer.shap_values(Xs)
            if isinstance(sv, list):
                local = np.asarray(sv[1]).reshape(-1)
            else:
                local = np.asarray(sv).reshape(-1)
                if local.size > len(_feature_cols):
                    local = local[: len(_feature_cols)]
            ranked = sorted(
                zip(_feature_cols, local, Xs.iloc[0].values),
                key=lambda t: abs(t[1]),
                reverse=True,
            )[:8]
            contributions = [
                {
                    "feature": f,
                    "shap": float(s),
                    "scaled_value": float(v),
                    "meaning": FEATURE_MEANINGS.get(feature_base(f), ""),
                }
                for f, s, v in ranked
            ]
    except Exception:
        if _importance is not None:
            for _, row in _importance.head(8).iterrows():
                f = row["feature"]
                contributions.append(
                    {
                        "feature": f,
                        "shap": float(row["mean_abs_shap"]),
                        "scaled_value": float(Xs.iloc[0][f]),
                        "meaning": FEATURE_MEANINGS.get(feature_base(f), ""),
                    }
                )

    return jsonify(
        {
            "prediction": pred,
            "label": label,
            "probability_malignant": proba,
            "probability_benign": 1.0 - proba,
            "model": _best_name,
            "contributions": contributions,
        }
    )


@app.route("/api/sample/<label>")
def sample(label: str):
    """Return a sample row from the dataset for quick demo."""
    from src.data import load_wdbc

    df = load_wdbc()
    target = 1 if label.lower().startswith("m") else 0
    row = df[df["diagnosis"] == target].iloc[0]
    payload = {c: float(row[c]) for c in FEATURE_NAMES if c in df.columns}
    return jsonify(payload)


if __name__ == "__main__":
    load_runtime()
    port = int(os.environ.get("PORT", "5000"))
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(host="0.0.0.0", port=port, debug=debug)
