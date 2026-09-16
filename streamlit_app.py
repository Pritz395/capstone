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
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Fraunces:opsz,wght@9..144,600;9..144,700&display=swap');

html, body, [class*="css"]  {
  font-family: "DM Sans", system-ui, sans-serif !important;
  color: #142033;
}

[data-testid="stAppViewContainer"] {
  background:
    radial-gradient(circle at 8% -10%, #ffd8bc 0%, transparent 42%),
    radial-gradient(circle at 95% 0%, #cfe8e2 0%, transparent 38%),
    #f3efe8;
}

[data-testid="stHeader"] { background: transparent; }
[data-testid="stToolbar"] { right: 0.6rem; }

.block-container {
  max-width: 1080px !important;
  padding-top: 1.1rem !important;
  padding-bottom: 2rem !important;
  padding-left: 1.25rem !important;
  padding-right: 1.25rem !important;
}

h1 {
  font-family: "Fraunces", Georgia, serif !important;
  font-size: clamp(1.65rem, 3vw, 2.2rem) !important;
  line-height: 1.15 !important;
  margin: 0 0 0.35rem 0 !important;
  color: #142033 !important;
  font-weight: 700 !important;
}

h2, h3, h4 {
  font-family: "Fraunces", Georgia, serif !important;
  color: #142033 !important;
}

.week-pill {
  display: inline-block;
  background: #e4f2f0;
  color: #0d5f5f;
  border: 1px solid #b7d7d2;
  border-radius: 999px;
  padding: 0.25rem 0.7rem;
  font-size: 0.8rem;
  font-weight: 600;
  margin-bottom: 0.65rem;
}

.sub {
  color: #5a6a7c;
  max-width: 62ch;
  line-height: 1.5;
  margin: 0 0 1rem 0;
  font-size: 0.98rem;
}

.metrics {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 0.65rem;
  margin: 0 0 0.85rem 0;
}
@media (max-width: 840px) {
  .metrics { grid-template-columns: 1fr 1fr; }
}
.metric {
  background: #fffcf7;
  border: 1px solid #d7cdc0;
  border-radius: 14px;
  padding: 0.7rem 0.85rem;
  box-shadow: 0 8px 20px rgba(20,32,51,0.035);
}
.metric .k { color: #5a6a7c; font-size: 0.76rem; }
.metric .v {
  font-family: "Fraunces", Georgia, serif;
  font-size: 1.28rem;
  margin-top: 0.12rem;
  color: #142033;
}

.steps {
  color: #5a6a7c;
  font-size: 0.9rem;
  background: #efe8de;
  border-radius: 12px;
  padding: 0.7rem 0.9rem;
  margin: 0 0 0.95rem 0;
  line-height: 1.45;
}
.steps strong { color: #142033; }

.panel {
  background: #fffcf7;
  border: 1px solid #d7cdc0;
  border-radius: 16px;
  padding: 0.95rem 1rem 0.75rem;
  box-shadow: 0 10px 24px rgba(20,32,51,0.035);
  margin-bottom: 0.9rem;
}
.panel-title {
  font-family: "Fraunces", Georgia, serif;
  font-size: 1.08rem;
  margin: 0 0 0.65rem 0;
  color: #142033;
}

.result-box {
  background: #f1ebe3;
  border-radius: 12px;
  padding: 0.95rem 1rem;
}
.label-pill {
  display: inline-block;
  padding: 0.32rem 0.68rem;
  border-radius: 999px;
  font-weight: 700;
  margin-bottom: 0.45rem;
  font-size: 0.92rem;
}
.benign { background: #d7f1e3; color: #1c7a4d; }
.malignant { background: #f7dcdc; color: #a43737; }
.muted { color: #5a6a7c; font-size: 0.85rem; }

.contrib { margin: 0.5rem 0; }
.contrib .dir { color: #5a6a7c; font-size: 0.82rem; }
.contrib .meaning { color: #5a6a7c; font-size: 0.78rem; margin-top: 0.08rem; }
.bar {
  height: 8px;
  border-radius: 999px;
  background: #e5ddd2;
  overflow: hidden;
  margin-top: 0.22rem;
}
.bar > span { display: block; height: 100%; background: #c45c2a; }

.foot {
  color: #5a6a7c;
  font-size: 0.82rem;
  margin-top: 0.4rem;
}

/* Buttons */
div.stButton > button {
  border-radius: 10px !important;
  font-weight: 650 !important;
  border: 0 !important;
  min-height: 2.55rem;
}
div.stButton > button[kind="primary"] {
  background: #0d5f5f !important;
  color: #fff !important;
}
div.stButton > button[kind="secondary"],
div.stButton > button:not([kind="primary"]) {
  background: #ebe3d8 !important;
  color: #142033 !important;
}

/* Compact number inputs — hide +/- steppers */
[data-testid="stNumberInput"] button { display: none !important; }
[data-testid="stNumberInput"] div[data-baseweb="input"] {
  background: #fff !important;
  border: 1px solid #d7cdc0 !important;
  border-radius: 8px !important;
}
[data-testid="stNumberInput"] label p {
  font-size: 0.72rem !important;
  color: #5a6a7c !important;
  font-weight: 500 !important;
}
[data-testid="stNumberInput"] input {
  padding-top: 0.35rem !important;
  padding-bottom: 0.35rem !important;
}

/* Scrollable feature grid inside panel */
.feature-scroll {
  max-height: 420px;
  overflow-y: auto;
  padding-right: 0.15rem;
}

[data-testid="stDataFrame"] {
  border: 1px solid #d7cdc0;
  border-radius: 12px;
  overflow: hidden;
  background: #fffcf7;
}

hr { display: none; }
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
    corpus_path = ARTIFACTS_DIR / "corpus" / "summary.csv"
    corpus_summary = pd.read_csv(corpus_path) if corpus_path.exists() else None
    return model, scaler, feature_cols, best_name, leaderboard, importance, corpus_summary


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
        return [
            {
                "feature": row["feature"],
                "shap": float(row["mean_abs_shap"]),
                "meaning": FEATURE_MEANINGS.get(feature_base(row["feature"]), ""),
            }
            for _, row in importance.head(8).iterrows()
        ]
    return []


def sample_row(kind: str) -> dict:
    df = load_wdbc()
    target = 1 if kind.startswith("m") else 0
    row = df[df["diagnosis"] == target].iloc[0]
    return {c: float(row[c]) for c in FEATURE_NAMES if c in df.columns}


def render_prediction(label: str, proba: float, best_name: str, contribs: list) -> None:
    pill = "malignant" if label == "Malignant" else "benign"
    st.markdown(
        f"""
        <div class="result-box">
          <div class="label-pill {pill}">{label}</div>
          <div><strong>Malignant probability:</strong> {proba*100:.2f}%</div>
          <div><strong>Benign probability:</strong> {(1-proba)*100:.2f}%</div>
          <div class="muted" style="margin-top:0.35rem">Model: {best_name}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if not contribs:
        return
    st.markdown("##### Why this prediction?")
    max_abs = max(abs(c["shap"]) for c in contribs) or 1.0
    parts = []
    for c in contribs:
        direction = "pushes malignant" if c["shap"] >= 0 else "pushes benign"
        width = min(abs(c["shap"]) / max_abs, 1.0) * 100
        meaning = c["meaning"] or ""
        parts.append(
            f"""
            <div class="contrib">
              <div><strong>{c['feature']}</strong> <span class="dir">({direction})</span></div>
              <div class="meaning">{meaning}</div>
              <div class="bar"><span style="width:{width}%"></span></div>
            </div>
            """
        )
    st.markdown("".join(parts), unsafe_allow_html=True)


def main():
    st.set_page_config(
        page_title="Capstone Demo — Breast Cancer XAI",
        page_icon="🩺",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)

    model, scaler, feature_cols, best_name, leaderboard, importance, corpus_summary = load_runtime()
    top = leaderboard.iloc[0] if leaderboard is not None and len(leaderboard) else None
    corpus_n = None
    if corpus_summary is not None and len(corpus_summary):
        all_row = corpus_summary[corpus_summary["dataset_id"] == "ALL"]
        if len(all_row):
            corpus_n = int(all_row.iloc[0]["n_samples"])

    # Seed widget state with a real sample (never show empty zeros)
    if "ui_ready" not in st.session_state:
        for col, val in sample_row("malignant").items():
            st.session_state[f"feat_{col}"] = float(val)
        st.session_state.auto_predict = True
        st.session_state.ui_ready = True

    st.markdown(
        '<div class="week-pill">Week 3–4 · WDBC live baseline · 2K corpus (not one merged train set)</div>',
        unsafe_allow_html=True,
    )
    st.markdown("# Explainable AI for Early Breast Cancer Detection")
    st.markdown(
        '<p class="sub">Live predictor uses the <strong>WDBC tabular</strong> baseline. '
        "A separate multi-dataset corpus (WDBC + WBCD + BUSI = 2,024 samples) is tracked with "
        "schema/modality provenance — WBCD/BUSI are <strong>not</strong> merged into this model’s training table.</p>",
        unsafe_allow_html=True,
    )

    acc = f"{top['accuracy']*100:.1f}%" if top is not None else "—"
    rec = f"{top['recall']*100:.1f}%" if top is not None else "—"
    auc = f"{top['roc_auc']:.3f}" if top is not None else "—"
    corpus_v = f"{corpus_n:,}" if corpus_n is not None else "—"
    st.markdown(
        f"""
        <div class="metrics">
          <div class="metric"><div class="k">Deployed model</div><div class="v">{best_name}</div></div>
          <div class="metric"><div class="k">Test accuracy</div><div class="v">{acc}</div></div>
          <div class="metric"><div class="k">Recall (malignancy)</div><div class="v">{rec}</div></div>
          <div class="metric"><div class="k">Corpus samples</div><div class="v">{corpus_v}</div></div>
        </div>
        <div class="steps"><strong>Demo in 15 seconds:</strong> a malignant sample loads automatically —
        switch with <strong>Load benign / malignant sample</strong>, or tweak values and click <strong>Predict</strong>.
        ROC-AUC on WDBC holdout: <strong>{auc}</strong>.</div>
        """,
        unsafe_allow_html=True,
    )

    if corpus_summary is not None:
        with st.expander("Multi-dataset corpus (2,024 samples — provenance layer)", expanded=False):
            st.caption(
                "Valid samples across modalities/schemas. This is NOT a single training dataframe. "
                "The live model above is trained on WDBC only. Feature schemas are never merged."
            )
            st.dataframe(corpus_summary, use_container_width=True, hide_index=True)

    b1, b2, b3 = st.columns(3)
    with b1:
        if st.button("Load benign sample", use_container_width=True):
            for col, val in sample_row("benign").items():
                st.session_state[f"feat_{col}"] = float(val)
            st.session_state.auto_predict = True
            st.rerun()
    with b2:
        if st.button("Load malignant sample", use_container_width=True):
            for col, val in sample_row("malignant").items():
                st.session_state[f"feat_{col}"] = float(val)
            st.session_state.auto_predict = True
            st.rerun()
    with b3:
        predict_clicked = st.button("Predict", type="primary", use_container_width=True)

    left, right = st.columns([1.35, 1], gap="medium")

    with left:
        st.markdown('<div class="panel"><div class="panel-title">1. Patient feature input</div>', unsafe_allow_html=True)
        values = {}
        cols = st.columns(3)
        for i, col in enumerate(feature_cols):
            with cols[i % 3]:
                values[col] = st.number_input(
                    col,
                    format="%.5f",
                    key=f"feat_{col}",
                )
        st.markdown("</div>", unsafe_allow_html=True)

    run = predict_clicked or st.session_state.pop("auto_predict", False)

    with right:
        st.markdown('<div class="panel"><div class="panel-title">2. Prediction & explanation</div>', unsafe_allow_html=True)
        if run:
            X = pd.DataFrame([[values[c] for c in feature_cols]], columns=feature_cols)
            Xs = pd.DataFrame(scaler.transform(X), columns=feature_cols)
            proba = float(model.predict_proba(Xs)[0][1])
            label = "Malignant" if proba >= 0.5 else "Benign"
            contribs = shap_contributions(model, best_name, feature_cols, Xs, importance)
            # Cache last result so layout stays filled after widget interactions
            st.session_state.last_result = {
                "label": label,
                "proba": proba,
                "contribs": contribs,
            }
            render_prediction(label, proba, best_name, contribs)
        elif "last_result" in st.session_state:
            r = st.session_state.last_result
            render_prediction(r["label"], r["proba"], best_name, r["contribs"])
        else:
            st.markdown(
                '<div class="result-box"><span class="muted">Load a sample, then click Predict.</span></div>',
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="panel"><div class="panel-title">3. Model comparison (held-out test set)</div>', unsafe_allow_html=True)
    if leaderboard is not None:
        show = leaderboard.copy()
        for col in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
            if col in show.columns:
                show[col] = show[col].map(lambda x: f"{x:.4f}")
        show.insert(0, "", show["model"].eq(best_name).map({True: "← deployed", False: ""}))
        st.dataframe(show, use_container_width=True, hide_index=True, height=320)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="panel"><div class="panel-title">4. Global SHAP importance</div>', unsafe_allow_html=True)
    if importance is not None:
        imp = importance.head(10).copy()
        if "mean_abs_shap" in imp.columns:
            imp["mean_abs_shap"] = imp["mean_abs_shap"].map(lambda x: f"{x:.4f}")
        st.dataframe(imp, use_container_width=True, hide_index=True, height=280)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        '<p class="foot">Academic decision-support prototype for CSE capstone review — not a clinical diagnostic device.</p>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
