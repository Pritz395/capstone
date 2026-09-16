# 12-Week Capstone Roadmap

**Project:** An Explainable AI Framework for Early Breast Cancer Detection and Classification  
**Scope (locked):** Wisconsin Diagnostic Breast Cancer (WDBC) tabular features → ML/DL models → metrics → XAI → web UI  
**Current status:** **Week 3 complete** + **2K corpus** + multimodal architecture locked in [`docs/ARCHITECTURE.md`](ARCHITECTURE.md)

---

## Outcome (end of Week 12)

A working system that:

1. Classifies tumors as **Benign / Malignant** from clinical nuclear features  
2. Compares multiple models and keeps the **best** by Accuracy / Recall / F1 / ROC-AUC  
3. Explains predictions with **SHAP** (and LIME)  
4. Exposes results through a simple **web interface**  
5. Is documented with literature, research gap, and a final report/demo  

---

## Week-by-week plan

| Week | Theme | Deliverables | Status |
|---|---|---|---|
| **1** | Problem & resources | Title/scope, resource inventory, paper folder, GitHub repo | **Done** |
| **2** | Data understanding | WDBC loaded, EDA, feature meanings, train/test strategy | **Done** |
| **3** | Baseline modeling | Multi-model training, metrics, leaderboard, best model saved | **Done** |
| **3b** | Corpus expansion | Multi-dataset registry → **2,024** valid samples (WDBC+WBCD+BUSI); schemas isolated | **Done** |
| **4** | Tuning & robustness | Hyperparameter search, cross-validation, class-imbalance checks | Planned |
| **5** | Deep tabular models | MLP refinement; optional LightGBM/CatBoost tuning bake-off | Planned |
| **6** | Explainability (SHAP) | Global + local SHAP reports tied to best model | Planned (code scaffold exists) |
| **7** | Explainability (LIME) + clinical narrative | LIME HTML, feature→medical meaning write-ups | Planned (code scaffold exists) |
| **8** | Web UI / demo | Flask (or similar) predict + explanation UI polish | Planned (MVP exists) |
| **9** | Extension track | Optional image path (Grad-CAM) **or** richer evaluation / fairness notes | Planned |
| **10** | Evaluation freeze | Final metrics tables, confusion/ROC figures, ablation summary | Planned |
| **11** | Report & docs | Capstone report draft, README finalization, ethics disclaimer | Planned |
| **12** | Demo & handoff | Presentation, recorded demo, clean release tag | Planned |

---

## What “best results” means here

We do **not** optimize accuracy alone.

| Metric | Why it matters |
|---|---|
| **Recall** | Missed malignancies (false negatives) are costly |
| **Precision** | Limits false alarms |
| **F1** | Balance of precision & recall |
| **ROC-AUC** | Ranking quality across thresholds |
| **Accuracy** | Overall correctness (reported, not sole criterion) |

Week 3 baseline (held-out 20% test, stratified):

- **Random Forest / SVM ≈ 97.37% accuracy**, recall ≈ 0.93, ROC-AUC ≈ 0.99  
- Deployed best for XAI-friendly path: **Random Forest**

---

## Folder map vs weeks

| Path | Weeks |
|---|---|
| `research/`, `papers/`, `docs/weeks/week-01.md` | 1 |
| `data/wdbc/`, `src/data.py`, `src/eda.py`, `docs/weeks/week-02.md` | 2 |
| `src/train.py`, `artifacts/leaderboard.csv`, `docs/weeks/week-03.md` | 3 |
| `src/explain.py`, `artifacts/shap_*`, `artifacts/lime_*` | 6–7 (early scaffold) |
| `app.py`, `templates/` | 8 (early MVP) |
| `docs/weeks/week-04.md` … `week-12.md` | Upcoming plans |

---

## Rules for the remaining weeks

1. One milestone commit (or PR) per week when possible  
2. Do not overwrite Week 3 baseline metrics — compare new runs in new artifact files  
3. Any image/Grad-CAM work is **optional** and must not break the WDBC tabular path  
4. UI always states: academic decision-support only, not clinical diagnosis  
