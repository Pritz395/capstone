"""Streamlit demo UI — styled to match the Flask capstone look (always-on host)."""

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

CSS = """
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&family=Fraunces:opsz,wght@9..144,600;9..144,700&display=swap');

html, body, [class*="css"] {
  font-family: "DM Sans", system-ui, sans-serif;
  color: #142033;
}

[data-testid="stAppViewContainer"] {
  background:
    radial-gradient(circle at 8% -10%, #ffd8bc 0%, transparent 42%),
    radial-gradient(circle at 95% 0%, #cfe8e2 0%, transparent 38%),
    #f3efe8;
}

[data-testid="stHeader"] {
  background: rgba(243, 239, 232, 0.85);
}

.block-container {
  max-width: 1100px;
  padding-top: 1.4rem;
  padding-bottom: 2rem;
}

h1, h2, h3, .fraunces {
  font-family: "Fraunces", Georgia, serif !important;
  color: #142033 !important;
}

.week-pill {
  display: inline-block;
  background: #e4f2f0;
  color: #0d5f5f;
  border: 1px solid #b7d7d2;
  border-radius: 999px;
  padding: 0.28rem 0.75rem;
  font-size: 0.82rem;
  font-weight: 600;
  margin-bottom: 0.55rem;
}

.sub {
  color: #5a6a7c;
  max-width: 62ch;
  line-height: 1.5;
  margin: 0.35rem 0 1rem;
}

.metrics {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 0.7rem;
  margin: 0.4rem 0 0.9rem;
}
@media (max-width: 800px) {
  .metrics { grid-template-columns: 1fr 1fr; }
}
.metric {
  background: #fffcf7;
  border: 1px solid #d7cdc0;
  border-radius: 14px;
  padding: 0.75rem 0.9rem;
  box-shadow: 0 8px 22px rgba(20,32,51,0.04);
}
.metric .k { color: #5a6a7c; font-size: 0.78rem; }
.metric .v {
  font-family: "Fraunces", Georgia, serif;
  font-size: 1.3rem;
  margin-top: 0.15rem;
  color: #142033;
}

.steps {
  color: #5a6a7c;
  font-size: 0.92rem;
  background: #efe8de;
  border-radius: 12px;
  padding: 0.75rem 0.95rem;
  margin-bottom: 0.85rem;
}
.steps strong { color: #142033; }

.card {
  background: #fffcf7;
  border: 1px solid #d7cdc0;
  border-radius: 16px;
  padding: 1rem 1.1rem 0.85rem;
  box-shadow: 0 10px 28px rgba(20,32,51,0.04);
  margin-bottom: 0.85rem;
}
.card h3 {
  font-family: "Fraunces", Georgia, serif;
  font-size: 1.12rem;
  margin: 0 0 0.7rem;
}

.result-box {
  background: #f1ebe3;
  border-radius: 12px;
  padding: 1rem;
  min-height: 96px;
}
.label-pill {
  display: inline-block;
  padding: 0.35rem 0.7rem;
  border-radius: 999px;
  font-weight: 700;
  margin-bottom: 0.45rem;
}
.benign { background: #d7f1e3; color: #1c7a4d; }
.malignant { background: #f7dcdc; color: #a43737; }

.contrib {
  margin: 0.55rem 0;
}
.contrib .dir { color: #5a6a7c; font-size: 0.85rem; }
.contrib .meaning { color: #5a6a7c; font-size: 0.8rem; margin-top: 0.1rem; }
.bar {
  height: 8px;
  border-radius: 999px;
  background: #e5ddd2;
  overflow: hidden;
  margin-top: 0.25rem;
}
.bar > span {
  display: block;
  height: 100%;
  background: #c45c2a;
}

.foot {
  color: #5a6a7c;
  font-size: 0.84rem;
  margin-top: 0.8rem;
}

div.stButton > button {
  border-radius: 10px;
  font-weight: 650;
  border: 0;
  padding: 0.55rem 0.9rem;
}
div.stButton > button[kind="primary"] {
  background: #0d5f5f;
  color: #fff;
}
div.stButton > button[kind="secondary"] {
  background: #ebe3d8;
  color: #142033;
}

[data-testid="stNumberInput"] label {
  font-size: 0.75rem !important;
  color: #5a6a7c !important;
}

[data-testid="stDataFrame"] {
  border: 1px solid #d7cdc0;
  border-radius: 12px;
  overflow: hidden;
}
"""


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


def inject_css() -> None:
    st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)


def main():
    st.set_page_config(
        page_title="Capstone Demo — Breast Cancer XAI",
        page_icon="🩺",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    inject_css()

    model, scaler, feature_cols, best_name, leaderboard, importance = load_runtime()
    top = leaderboard.iloc[0] if leaderboard is not None and len(leaderboard) else None

    st.markdown('<div class="week-pill">Week 3 review · Capstone demo UI</div>', unsafe_allow_html=True)
    st.markdown("# Explainable AI for Early Breast Cancer Detection")
    st.markdown(
        '<p class="sub">Live demo of our trained classifier on Wisconsin Diagnostic Breast Cancer '
        "(WDBC) features: predict Benign vs Malignant, then show which features drove the decision (SHAP).</p>",
        unsafe_allow_html=True,
    )

    acc = f"{top['accuracy']*100:.1f}%" if top is not None else "—"
    rec = f"{top['recall']*100:.1f}%" if top is not None else "—"
    auc = f"{top['roc_auc']:.3f}" if top is not None else "—"
    st.markdown(
        f"""
        <div class="metrics">
          <div class="metric"><div class="k">Deployed model</div><div class="v">{best_name}</div></div>
          <div class="metric"><div class="k">Test accuracy</div><div class="v">{acc}</div></div>
          <div class="metric"><div class="k">Recall (malignancy)</div><div class="v">{rec}</div></div>
          <div class="metric"><div class="k">ROC-AUC</div><div class="v">{auc}</div></div>
        </div>
        <div class="steps"><strong>Demo in 15 seconds:</strong> click <strong>Load malignant sample</strong>
        → read the label, probability, and “Why this prediction?” Then try a benign sample.</div>
        """,
        unsafe_allow_html=True,
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

    left, right = st.columns([1.35, 1], gap="medium")

    with left:
        st.markdown('<div class="card"><h3>1. Patient feature input</h3></div>', unsafe_allow_html=True)
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
        st.markdown('<div class="card"><h3>2. Prediction & explanation</h3>', unsafe_allow_html=True)
        if run:
            X = pd.DataFrame([[values[c] for c in feature_cols]], columns=feature_cols)
            Xs = pd.DataFrame(scaler.transform(X), columns=feature_cols)
            proba = float(model.predict_proba(Xs)[0][1])
            label = "Malignant" if proba >= 0.5 else "Benign"
            pill = "malignant" if label == "Malignant" else "benign"
            st.markdown(
                f"""
                <div class="result-box">
                  <div class="label-pill {pill}">{label}</div>
                  <div><strong>Malignant probability:</strong> {proba*100:.2f}%</div>
                  <div><strong>Benign probability:</strong> {(1-proba)*100:.2f}%</div>
                  <div class="dir" style="margin-top:0.35rem;color:#5a6a7c;font-size:0.85rem">Model: {best_name}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            contribs = shap_contributions(model, best_name, feature_cols, Xs, importance)
            if contribs:
                st.markdown("#### Why this prediction?")
                max_abs = max(abs(c["shap"]) for c in contribs) or 1.0
                html = []
                for c in contribs:
                    direction = "pushes malignant" if c["shap"] >= 0 else "pushes benign"
                    width = min(abs(c["shap"]) / max_abs, 1.0) * 100
                    meaning = c["meaning"] or ""
                    html.append(
                        f"""
                        <div class="contrib">
                          <div><strong>{c['feature']}</strong> <span class="dir">({direction})</span></div>
                          <div class="meaning">{meaning}</div>
                          <div class="bar"><span style="width:{width}%"></span></div>
                        </div>
                        """
                    )
                st.markdown("".join(html), unsafe_allow_html=True)
        else:
            st.markdown(
                '<div class="result-box"><span style="color:#5a6a7c">Load a sample, then click Predict.</span></div>',
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="card"><h3>3. Model comparison (held-out test set)</h3></div>', unsafe_allow_html=True)
    if leaderboard is not None:
        show = leaderboard.copy()
        show["deployed"] = show["model"].eq(best_name).map({True: "← deployed", False: ""})
        st.dataframe(show, use_container_width=True, hide_index=True)
    else:
        st.write("Leaderboard not found.")

    st.markdown('<div class="card"><h3>4. Global SHAP importance</h3></div>', unsafe_allow_html=True)
    if importance is not None:
        st.dataframe(importance.head(10), use_container_width=True, hide_index=True)

    st.markdown(
        '<p class="foot">Academic decision-support prototype for CSE capstone review — not a clinical diagnostic device.</p>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
