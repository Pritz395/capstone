# Week 2 — Dataset understanding & EDA

**Status:** Complete

## Goals

- Load and validate WDBC  
- Understand every feature family medically enough for the report  
- Define train/test split + scaling strategy  

## Dataset facts

| Item | Value |
|---|---|
| Name | Wisconsin Diagnostic Breast Cancer (WDBC) |
| Samples | 569 |
| Features | 30 real-valued nuclear morphometry features |
| Labels | Benign (357) / Malignant (212) |
| Missing values | None |
| Source files | `data/wdbc/wdbc.data`, `wdbc.names`, `breast-cancer.csv` |

Features = 10 nucleus measures × {mean, SE, worst}:  
radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, fractal dimension.

## Done this week

- `src/data.py` — load, clean, stratified split, `StandardScaler`  
- `src/eda.py` — class balance, summary stats, correlation peek, plots → `artifacts/eda/`  
- Feature meaning table in `research/01_PROJECT_NOTES.md`  

## Split protocol (locked for Week 3+)

- Stratified **80 / 20** train/test  
- `random_state=42`  
- Scale with train-fit `StandardScaler` only  

## Exit criteria (met)

- [x] Dataset loads reproducibly  
- [x] Class distribution documented  
- [x] Feature dictionary drafted  
- [x] Preprocessing pipeline code complete  
