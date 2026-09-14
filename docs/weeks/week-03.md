# Week 3 — Baseline models & metrics

**Status:** Complete — **this is the current push milestone**

## Goals

- Train multiple classifiers on WDBC  
- Compare with medically relevant metrics  
- Select a best model for later XAI / UI weeks  

## Models trained

Logistic Regression, Decision Tree, Random Forest, SVM, KNN, XGBoost, LightGBM, CatBoost, MLP

## Held-out test results

See `artifacts/leaderboard.csv`.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| SVM | 0.9737 | 1.000 | 0.9286 | 0.9630 | 0.9947 |
| **Random Forest** | **0.9737** | **1.000** | **0.9286** | **0.9630** | **0.9944** |
| CatBoost | 0.9649 | 1.000 | 0.9048 | 0.9500 | 0.9990 |
| LightGBM | 0.9649 | 1.000 | 0.9048 | 0.9500 | 0.9970 |
| Logistic Regression | 0.9649 | 0.975 | 0.9286 | 0.9512 | 0.9960 |
| XGBoost | 0.9649 | 1.000 | 0.9048 | 0.9500 | 0.9950 |
| MLP | 0.9649 | 1.000 | 0.9048 | 0.9500 | 0.9927 |
| KNN | 0.9561 | 0.974 | 0.9048 | 0.9383 | 0.9823 |
| Decision Tree | 0.9298 | 0.905 | 0.9048 | 0.9048 | 0.9246 |

**Deployed best model:** Random Forest (tied top accuracy with SVM; preferred for fast tree-based SHAP in later weeks).

## Artifacts

- `src/train.py`  
- `artifacts/leaderboard.csv`  
- `artifacts/metrics.json`  
- `artifacts/confusion_matrices.png`  
- `artifacts/roc_curves.png`  
- `models/best_model_name.txt` → `Random Forest`  

## How to reproduce

```bash
source .venv/bin/activate
python -m src.train
```

## Exit criteria (met)

- [x] ≥5 models compared  
- [x] Accuracy, Precision, Recall, F1, ROC-AUC reported  
- [x] Confusion matrices + ROC curves saved  
- [x] Best model selection rule documented  
- [x] Results in the literature ballpark (~97%+ on WDBC)  

## Next (Week 4)

Hyperparameter tuning + cross-validation without losing this baseline for comparison.  
