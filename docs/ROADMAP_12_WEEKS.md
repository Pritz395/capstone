# 12-Week Capstone Roadmap

**Project:** An Explainable AI Framework for Early Breast Cancer Detection and Classification  
**End goal:** Multimodal (tabular + image) explainable framework — see [`ARCHITECTURE.md`](ARCHITECTURE.md)  
**Current checkpoint:** **Week 3–4 complete** (foundation, not final fusion)

---

## Clean progression

```
WEEK 1  Problem + literature
   ↓
WEEK 2  WDBC + EDA
   ↓
WEEK 3  WDBC baseline + WBCD + 2K multi-dataset corpus
   ↓
WEEK 4  Modality-aware architecture (tabular/image separation)
   ↓
WEEK 5  CNN / image experiments (BUSI)
   ↓
WEEK 6–7  Representation + missing-modality fusion
   ↓
WEEK 8+   20K+ expansion + tuning + XAI
   ↓
FINAL   One unified prediction framework
```

---

## Week status

| Week | Theme | Status |
|---|---|---|
| **1** | Problem + literature | **Done** |
| **2** | WDBC + EDA | **Done** |
| **3** | WDBC + WBCD baselines; 2,024 corpus; provenance | **Done** |
| **4** | Modality-aware structure + image skeletons | **Done** |
| **5** | Image/CNN experiments on BUSI | Planned |
| **6–7** | Latent fusion (missing-modality aware) | Planned |
| **8+** | BreakHis / 20K+, tuning, Grad-CAM + SHAP polish | Planned |
| **11–12** | Report, demo, handoff | Planned |

---

## Week 3–4 checkpoint (what is true now)

- **Corpus:** 2,024 samples across WDBC + WBCD + BUSI (**not** one merged training table)
- **Working predictor:** tabular WDBC baselines (+ live demo)
- **New:** independent WBCD tabular baselines
- **Architecture:** tabular vs image pipelines separated; image encoder is a **skeleton**
- **Not yet:** CNN training, fusion, BreakHis, final unified UI

---

## Reproduce checkpoint

```bash
python scripts/fetch_datasets.py --all
python scripts/build_corpus.py --min-samples 2000
python scripts/validate_corpus.py
python -m src.train_tabular --dataset all
```
