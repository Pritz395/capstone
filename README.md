# Explainable AI Framework for Early Breast Cancer Detection

**12-week CSE capstone** · [Pritz395/capstone](https://github.com/Pritz395/capstone)  
**Current checkpoint:** **Week 3–4** (accurate to the code)

> Academic decision-support prototype only — **not** a clinical diagnostic device.

## What is true right now

| Layer | Status |
|---|---|
| **Trained ML** | **WDBC tabular** (9 models compared; best ≈ **97.4%** acc) |
| **Also trained** | **WBCD tabular** baselines (separate 9-feature schema) |
| **Multi-dataset corpus** | **2,024** samples = WDBC 569 + WBCD 675 + BUSI 780 |
| **BUSI** | Registered in corpus / provenance only — **no CNN trained yet** |
| **Multimodal fusion** | Designed (`docs/ARCHITECTURE.md`) — **not implemented** |

**Say this to mentors:**  
“We established a **2,024-sample multi-dataset corpus** with schema/modality separation. **WDBC remains the deployed live baseline.** WBCD and BUSI are integrated into the dataset/provenance layer and will feed later multimodal work.”

**Do not say:** “Our model is trained on all 2,024 samples.”

## Live demo

https://capstone-jjgpitxnzxxtmljc6fhs75.streamlit.app/  
(WDBC predictor + corpus summary)

## Current WDBC results (held-out 20%)

| Best models | Accuracy | Recall | ROC-AUC |
|---|---:|---:|---:|
| Random Forest / SVM | **97.37%** | 92.86% | ~0.994 |

Full tables: `artifacts/wdbc/leaderboard.csv`, `artifacts/wbcd/leaderboard.csv`

## Repo layout (Week 3–4)

```
data/wdbc|wbcd|busi
configs/datasets.yaml
src/datasets/          # registry + loaders (schemas isolated)
src/preprocessing/     # tabular + image skeleton
src/models/            # tabular zoo + image encoder skeleton
src/train_tabular.py   # train WDBC or WBCD independently
artifacts/corpus/      # 2024-row provenance manifest
artifacts/wdbc|wbcd     # per-schema metrics
docs/ARCHITECTURE.md   # future multimodal design
docs/weeks/week-03.md
docs/weeks/week-04.md
streamlit_app.py
```

## Progression

```
Week 1–2  Problem, WDBC, EDA
Week 3    WDBC baseline + WBCD + 2K corpus
Week 4    Modality-aware structure (tabular vs image skeletons)
Week 5+   BUSI CNN → fusion → 20K+ → unified XAI UI
```

## Quick start (developers)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt

python scripts/fetch_datasets.py --all
python scripts/build_corpus.py --min-samples 2000
python scripts/validate_corpus.py
python -m src.train_tabular --dataset all

streamlit run streamlit_app.py
```

Details: [`docs/DATASETS.md`](docs/DATASETS.md) · [`docs/ROADMAP_12_WEEKS.md`](docs/ROADMAP_12_WEEKS.md)
