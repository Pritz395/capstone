# Resource inventory

Everything gathered for the capstone, mapped to how we use it.

## Datasets (local)

| File | What it is | Use |
|---|---|---|
| `data/wdbc/wdbc.data` + `wdbc.names` | Official UCI WDBC (569 × 30 features) | **Primary training data** |
| `data/wdbc/breast-cancer.csv` | Same WDBC with headers | Used by `src/data.py` |
| `data/wdbc/kaggle/data.csv` | Kaggle export of WDBC | Duplicate; keep for reference |

**Decision:** Train on **WDBC tabular**. Fits a CSE capstone, matches several of your papers, and enables SHAP cleanly.

### WDBC facts (from `wdbc.names`)

- 569 instances; **357 benign**, **212 malignant**
- Features from fine-needle aspirate (FNA) cell nuclei
- 10 nucleus measures × (mean, SE, worst) = 30 features
- No missing values
- Classes: B / M

## Local papers (`papers/`)

| PDF | Title (short) | Relevance |
|---|---|---|
| `An_Explainable_Artificial_Intelligence_Model_for_t.pdf` | IEEE Access XAI classification of breast cancer (Khater et al.) | Closest template: WDBC + ML + model-agnostic XAI; reports ~97.7–98.6% |
| `1-s2.0-S2772442524000558-main.pdf` | Healthcare Analytics: ML + LASSO/SHAP feature selection | SHAP for feature selection + high accuracy on breast cancer tabular data |
| `s41598-025-97718-5.pdf` | Scientific Reports: DNBCD on histopathology + ultrasound + Grad-CAM | Image/XAI path if you extend beyond WDBC later |
| `An_Explainable_Artificial_Intelligence_Framework_f.pdf` | IJEEEMI 2025: XAI framework for breast cancer (Ridha et al.) | Thermal imaging (DMR): Attention U-Net + K-Means + EfficientNet-B7 + **LIME**; val acc ~91.7% — cite for LIME / image-framework comparison |
| `fimmu-16-1658741.pdf` | Frontiers in Immunology 2025: hybrid DL + XAI (Zou & Miao) | Ultrasound: fused DenseNet121 + Xception + VGG16 + **Grad-CAM++**; ~97% accuracy — cite for Week 9 image/Grad-CAM extension |

## Online papers / reviews

| Link | Role |
|---|---|
| https://pubmed.ncbi.nlm.nih.gov/39430216/ | **Core XAI review** — SHAP dominant in breast-cancer XAI |
| https://pubmed.ncbi.nlm.nih.gov/37278831/ | ML/DL imaging modalities overview (mammo, US, MRI, histology, thermo) |
| https://pubmed.ncbi.nlm.nih.gov/36103745/ | DL datasets, methods, challenges |
| https://pubmed.ncbi.nlm.nih.gov/39403286/ | Multimodal + XAI (Grad-CAM, SHAP, LIME) review |
| https://pubmed.ncbi.nlm.nih.gov/38517775/ | Related PubMed entry (fetch blocked earlier; keep for literature survey) |
| https://www.mdpi.com/2227-9032/12/10/1025 | MDPI Healthcare paper (access blocked from this environment) |
| https://onlinelibrary.wiley.com/doi/10.1002/cai2.136 | Wiley review (Cloudflare challenge) |

## Background / data portals

| Link | Role |
|---|---|
| https://www.who.int/news-room/fact-sheets/detail/breast-cancer | Medical background (incidence, risk, WHO) |
| https://ohsl.us/projects/bcda | Breast Cancer Data Alliance catalog / big-data context |
| https://cdas.cancer.gov/datasets/plco/19/ | PLCO breast datasets — **application required**; not for quick student baseline |

## Recommended framework (locked)

```
WDBC data
  → clean / scale
  → Logistic Regression, DT, RF, SVM, KNN, XGBoost, LightGBM, CatBoost, MLP
  → metrics (Acc, Prec, Rec, F1, ROC-AUC, CM)
  → best model
  → SHAP (global + local) + LIME
  → Flask UI (prediction + explanation)
```

## Research gap angles (for report)

1. Many high-accuracy models are **black boxes** — clinicians need feature-level reasons.
2. Accuracy alone is insufficient; **recall** matters (missed malignancies).
3. XAI visuals are common, but **clinical validation of explanations** is still weak (reviews emphasize this).
4. Student systems often stop at “98% accuracy”; this project adds **comparative models + XAI + UI**.
