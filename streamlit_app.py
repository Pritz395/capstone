"""Streamlit demo UI — free host on Streamlit Community Cloud (no card)."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from src.data import FEATURE_MEANINGS, FEATURE_NAMES, load_wdbc

ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
ARTIFACTS_DIR = ROOT / "artifacts"


@st.cache_resource
def load_runtime():
    model = joblib.load(MODELS_DIR / "best_model.joblib")
    scaler = joblib.load(MODELS_DIR / "scaler.joblib")
    feature_cols = joblib.load(MODELS_DIR / "feature_cols.joblib")
    name_path = MODELS_DIR / "best_model_name.txt"
    best_name = name_path.read_text().strip() if name_path.exists() else "Random Forest"
    lb_path = ARTIFACTS_DIR / "leaderboard.csv"
    leaderboard = pd.read_csv(lb_path) if lb_path.exists() else None
    imp_path = ARTIFACTS_DIR / "shap_feature_importance.csv"
    importance = pd.read_csv(imp_path) if imp_path.exists() else None
    return model, scaler, feature_cols, best_name, leaderboard, importance


def feature_base(name: str) -> str:
    for key in FEATURE_MEANINGS:
        if name.startswith(key):
            return key
    return name


def shap_contributions(model, best_name, feature_cols, Xs, importance):
    try:
        import shap

        tree_names = {"Random Forest", "Decision Tree", "XGBoost", "LightGBM", "CatBoost"}
        if best_name in tree_names:
            explainer = shap.TreeExplainer(model)
            sv = explainer.shap_values(Xs)
            if isinstance(sv, list):
                local = np.asarray(sv[1]).reshape(-1)
            else:
                local = np.asarray(sv).reshape(-1)
                if local.size > len(feature_cols):
                    local = local[: len(feature_cols)]
            ranked = sorted(
                zip(feature_cols, local),
                key=lambda t: abs(t[1]),
                reverse=True,
            )[:8]
            return [
                {
                    "feature": f,
                    "shap": float(s),
                    "meaning": FEATURE_MEANINGS.get(feature_base(f), ""),
                }
                for f, s in ranked
            ]
    except Exception:
        pass

    if importance is not None:
        rows = []
        for _, row in importance.head(8).iterrows():
            rows.append(
                {
                    "feature": row["feature"],
                    "shap": float(row["mean_abs_shap"]),
                    "meaning": FEATURE_MEANINGS.get(feature_base(row["feature"]), ""),
                }
            )
        return rows
    return []


def sample_row(kind: str) -> dict:
    df = load_wdbc()
    target = 1 if kind.startswith("m") else 0
    row = df[df["diagnosis"] == target].iloc[0]
    return {c: float(row[c]) for c in FEATURE_NAMES if c in df.columns}


def main():
    st.set_page_config(
        page_title="Capstone Demo — Breast Cancer XAI",
        page_icon="🩺",
        layout="wide",
    )

    model, scaler, feature_cols, best_name, leaderboard, importance = load_runtime()

    st.caption("Week 3 review · Capstone demo UI")
    st.title("Explainable AI for Early Breast Cancer Detection")
    st.write(
        "Live classifier on Wisconsin Diagnostic Breast Cancer (WDBC) features: "
        "predict **Benign vs Malignant**, then show which features drove the decision (SHAP)."
    )

    top = leaderboard.iloc[0] if leaderboard is not None and len(leaderboard) else None
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Deployed model", best_name)
    c2.metric("Test accuracy", f"{top['accuracy']*100:.1f}%" if top is not None else "—")
    c3.metric("Recall", f"{top['recall']*100:.1f}%" if top is not None else "—")
    c4.metric("ROC-AUC", f"{top['roc_auc']:.3f}" if top is not None else "—")

    st.info(
        "**Demo in 15 seconds:** click **Load malignant sample** → read the prediction → "
        "try a benign sample."
    )

    if "form_values" not in st.session_state:
        st.session_state.form_values = {c: 0.0 for c in feature_cols}

    b1, b2, b3 = st.columns([1, 1, 1])
    with b1:
        if st.button("Load benign sample", use_container_width=True):
            st.session_state.form_values = sample_row("benign")
            st.session_state.auto_predict = True
    with b2:
        if st.button("Load malignant sample", use_container_width=True):
            st.session_state.form_values = sample_row("malignant")
            st.session_state.auto_predict = True
    with b3:
        predict_clicked = st.button("Predict", type="primary", use_container_width=True)

    left, right = st.columns([1.35, 1])

    with left:
        st.subheader("1. Patient feature input")
        values = {}
        cols = st.columns(3)
        for i, col in enumerate(feature_cols):
            with cols[i % 3]:
                values[col] = st.number_input(
                    col,
                    value=float(st.session_state.form_values.get(col, 0.0)),
                    format="%.5f",
                    key=f"feat_{col}",
                )
        st.session_state.form_values = values

    run = predict_clicked or st.session_state.pop("auto_predict", False)

    with right:
        st.subheader("2. Prediction & explanation")
        if run:
            X = pd.DataFrame([[values[c] for c in feature_cols]], columns=feature_cols)
            Xs = pd.DataFrame(scaler.transform(X), columns=feature_cols)
            proba = float(model.predict_proba(Xs)[0][1])
            label = "Malignant" if proba >= 0.5 else "Benign"
            if label == "Malignant":
                st.error(f"**{label}** — malignant probability {proba*100:.2f}%")
            else:
                st.success(f"**{label}** — malignant probability {proba*100:.2f}%")
            st.caption(f"Model: {best_name} · benign probability {(1-proba)*100:.2f}%")

            contribs = shap_contributions(model, best_name, feature_cols, Xs, importance)
            if contribs:
                st.markdown("#### Why this prediction?")
                max_abs = max(abs(c["shap"]) for c in contribs) or 1.0
                for c in contribs:
                    direction = "pushes malignant" if c["shap"] >= 0 else "pushes benign"
                    st.write(f"**{c['feature']}** — {direction}")
                    if c["meaning"]:
                        st.caption(c["meaning"])
                    st.progress(min(abs(c["shap"]) / max_abs, 1.0))
        else:
            st.write("Load a sample, then click Predict.")

    st.subheader("3. Model comparison (held-out test set)")
    if leaderboard is not None:
        show = leaderboard.copy()
        show["deployed"] = show["model"].eq(best_name).map({True: "← deployed", False: ""})
        st.dataframe(show, use_container_width=True, hide_index=True)
    else:
        st.write("Leaderboard not found.")

    st.subheader("4. Global SHAP importance")
    if importance is not None:
        st.dataframe(importance.head(10), use_container_width=True, hide_index=True)

    st.caption(
        "Academic decision-support prototype for CSE capstone review — not a clinical diagnostic device."
    )


if __name__ == "__main__":
    main()
