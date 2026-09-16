# Week 3 — Expanded data + baseline ML

**Status:** Complete (Week 3–4 checkpoint)

## Story

WDBC baseline → expanded corpus with WBCD + BUSI (images registered only) → multi-model evaluation on **each tabular schema separately** → **2,024-sample multi-dataset corpus** (not one merged training table).

## Done

- WDBC pipeline remains working (demo + baselines)
- WBCD added as second **tabular** dataset (9 ordinal features; schema ≠ WDBC)
- BUSI registered as **image** modality in the corpus (no CNN trained yet)
- Dataset registry + provenance manifest (`artifacts/corpus/`)
- Validation: no fake schema merging, no fabricated pairs
- Independent model comparison on WDBC and on WBCD
- Metrics: Accuracy, Precision, Recall, Specificity, F1, ROC-AUC, confusion matrices

## Dataset counts (corpus)

| Dataset | Modality | N |
|---|---|---:|
| WDBC | tabular | 569 |
| WBCD | tabular | 675 |
| BUSI | image | 780 |
| **ALL (corpus)** | mixed | **2,024** |

## Important wording

**Accurate mentor line:**  
“2,024-sample multi-modal corpus established; **WDBC remains the current trained baseline**; WBCD and BUSI are integrated into the dataset/provenance layer and will feed the subsequent multimodal ML/DL pipeline.”

Say: **“multi-dataset corpus: 2,024 samples.”**  
Do **not** say: “one model trained on 2,024 unified rows.”

## Reproduce

```bash
python scripts/build_corpus.py --min-samples 2000
python -m src.train_tabular --dataset all
```

Artifacts:
- `artifacts/wdbc/` — WDBC leaderboard/metrics/plots
- `artifacts/wbcd/` — WBCD leaderboard/metrics/plots
- `artifacts/corpus/` — full provenance manifest
