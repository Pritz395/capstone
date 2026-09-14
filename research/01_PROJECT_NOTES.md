# Capstone research notes (CSE scope)

## 1. Problem definition (locked)

**Task:** Binary classification of breast tumors as **Benign (0)** or **Malignant (1)** from **30 FNA nuclear morphometry features** (WDBC).

**Not in v1:** mammogram detection bounding boxes, histopathology CNNs, multimodal fusion.

## 2. Why AI here

- Early triage support from structured clinical/cytology features
- Consistent second opinion alongside standard diagnostics
- Explanations (SHAP) surface *which* features drove a call

## 3. Feature meanings (short)

| Family | Meaning |
|---|---|
| radius | Mean distance from nucleus center to perimeter |
| texture | Gray-level SD inside nucleus |
| perimeter / area | Size of nucleus |
| smoothness | Local radius variation |
| compactness | `perimeter²/area − 1` |
| concavity / concave points | Indentations of contour (often linked to irregular malignant nuclei) |
| symmetry | Shape symmetry |
| fractal dimension | Contour complexity |

Each exists as **mean**, **standard error**, and **worst** (largest) → 30 inputs.

## 4. Models we compare

Traditional ML: LR, DT, RF, SVM, KNN, XGBoost (+ LightGBM, CatBoost to match abstract).  
Deep (tabular): MLP.

## 5. Metrics that matter medically

- **Recall (sensitivity):** catch malignancies — prioritize discussing FN
- **Precision:** control false alarms
- **ROC-AUC:** ranking quality across thresholds
- Accuracy alone is not enough for the report write-up

## 6. XAI

- **SHAP:** global importance + local contributions (model-agnostic / tree-native)
- **LIME:** local surrogate explanation HTML for one sample
- Grad-CAM reserved for future image models

## 7. Ethics line for report / UI

Prototype for coursework and research demonstration. Not FDA/CE cleared. Always human-in-the-loop.
