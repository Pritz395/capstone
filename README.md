# Explainable AI Framework for Early Breast Cancer Detection

**12-week CSE capstone** · Repo: [Pritz395/capstone](https://github.com/Pritz395/capstone)  
**Current milestone:** Week 3 — baseline models & metrics (**pushed**)

A tabular ML system on the **Wisconsin Diagnostic Breast Cancer (WDBC)** dataset that compares multiple models, reports medically relevant metrics, and (in later weeks) explains predictions with SHAP/LIME via a web UI.

> Academic decision-support prototype only — **not** a clinical diagnostic device.

## Current results (Week 3)

Held-out stratified test set (20%):

| Best models | Accuracy | Recall | ROC-AUC |
|---|---:|---:|---:|
| Random Forest / SVM | **97.37%** | 92.86% | ~0.994 |

Full table: [`artifacts/leaderboard.csv`](artifacts/leaderboard.csv)  
Roadmap: [`docs/ROADMAP_12_WEEKS.md`](docs/ROADMAP_12_WEEKS.md)

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt   # local training stack

python -m src.eda      # Week 2 — exploratory analysis
python -m src.train    # Week 3 — train & compare models
python -m src.explain  # Week 6–7 scaffold — SHAP / LIME
streamlit run streamlit_app.py   # demo UI (also deployed on Streamlit Cloud)
# or: python app.py              # Flask UI @ http://127.0.0.1:5000
```

**Always-on demo:** see [`docs/DEPLOY.md`](docs/DEPLOY.md) (Streamlit Community Cloud — free, no card).

## Repo layout

```
docs/ROADMAP_12_WEEKS.md   # full 12-week plan
docs/weeks/                # per-week status (01–03 done)
data/wdbc/                 # WDBC dataset
papers/                    # gathered PDFs
research/                  # literature inventory & notes
src/data.py                # load / split / scale
src/eda.py                 # EDA
src/train.py               # multi-model training
src/explain.py             # SHAP + LIME (later weeks)
app.py + templates/        # UI MVP (later weeks)
artifacts/                 # metrics, plots, explanations
```

## 12-week status (short)

| Weeks | Focus | Status |
|---|---|---|
| 1 | Problem, papers, resources | Done |
| 2 | Data + EDA | Done |
| 3 | Baseline models + metrics | **Done (this push)** |
| 4–5 | Tuning + MLP bake-off | Next |
| 6–8 | SHAP / LIME / UI polish | Planned (early code present) |
| 9–12 | Extension, freeze, report, demo | Planned |
