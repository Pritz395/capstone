# Architecture Design — Explainable Multimodal Breast Cancer Framework

**Project title:** An Explainable AI Framework for Early Breast Cancer Detection and Classification Using Advanced Machine Learning and Deep Learning Algorithms  

**Document type:** Design only (no major implementation in this step)  
**Status date:** aligned with current repo (WDBC + WBCD + BUSI corpus ≈ **2,024** samples)  
**Principle:** One user-facing **Benign / Malignant** framework; internally modality-aware; never fake-merge incompatible raw features.

---

## 0. Correction relative to the current repo

| Current reality | Required correction |
|---|---|
| Live predictor is **WDBC-tabular Random Forest only** | Final system must support **tabular and images** under one framework |
| Corpus treats WDBC / WBCD / BUSI as isolated catalogs | Keep isolation of **raw schemas**, but add **encoders → latent → fusion → one decision** |
| 2K / 20K framed as the goal | Sample growth is a **means**; the goal is a **multimodal XAI detection/classification framework** |
| Independent models “forever” | Wrong end-state. Independent encoders are **internal components**, not the product |

This document redesigns toward that end-state while remaining academically honest about **unpaired** public datasets.

---

## 1. How each dataset enters the system

```
                    EXTERNAL DATASETS
                            |
        +-------------------+-------------------+
        |                   |                   |
     WDBC (569)         WBCD (675)          BUSI (780)
   30 morphometry     9 ordinal scores     ultrasound PNG
   schema A             schema B            schema C
        |                   |                   |
        v                   v                   v
   tabular adapter A   tabular adapter B   image loader
        |                   |                   |
        +--------+----------+                   |
                 |                              |
                 v                              v
           TABULAR BRANCH                  IMAGE BRANCH
                 |                              |
                 +--------------+---------------+
                                |
                         FUSION / HEAD
                                |
                      Benign / Malignant
```

| Dataset | Modality gate | Raw representation | Adapter | Can train fusion with paired patient record? |
|---|---|---|---|---|
| **WDBC** | Tabular | 30 continuous FNA features | Schema-A tabular encoder | No paired image in this dataset |
| **WBCD** | Tabular | 9 ordinal cytology scores | Schema-B tabular encoder | No paired image |
| **BUSI** | Image | Ultrasound image (+ optional mask) | Vision encoder | No paired WDBC/WBCD row |
| **BreakHis** (future) | Image | Histopathology tiles | Vision encoder (same branch, possibly domain tag) | Unpaired |
| Future clinical CSVs | Tabular | Must declare `schema_id` | New adapter or reject | Only if truly paired |

**Corpus role:** `artifacts/corpus/manifest.csv` remains the **provenance spine** (`sample_uid`, `dataset_id`, `schema_id`, `modality`, labels).  
**Training role:** samples feed the branch matching their modality; they do **not** become one giant dataframe.

---

## 2. Tabular encoder (what it should be)

### 2.1 Problem
WDBC and WBCD are both “tabular” but **not the same feature space**. Concatenating them is invalid.

### 2.2 Capstone-defensible design: **schema adapters → shared tabular latent**

```
raw tabular (schema A or B or future Z)
        |
        v
 schema-specific adapter
  (linear / MLP that only sees that schema’s columns)
        |
        v
 shared tabular latent z_tab  ∈ R^d
        |
        v
 (optional) tabular classifier head for unimodal training
```

**Recommended concrete choice for CSE scope:**

1. **Per-schema adapter MLP**  
   - WDBC: `30 → 64 → d`  
   - WBCD: `9 → 64 → d`  
   - Future schema: new adapter, same `d`
2. **Shared tabular trunk** (small MLP) refining `z_tab`
3. Classical models (RF, XGBoost, …) remain available as:
   - **baselines / ablations** on a single schema, and/or
   - **teachers** that do not define the final fused API

**Why not one Random Forest on all tabular rows?**  
Because columns differ; forcing one RF would require fabrication or meaningless padding.

**Model selection (tabular branch evaluation set):**  
Logistic Regression, SVM, Random Forest, XGBoost, LightGBM, CatBoost, MLP — compare **within each schema**, then decide which adapter/trunk recipe advances to fusion (likely MLP encoder + gradient-boosted or MLP head, chosen by validation—not assumed RF).

---

## 3. Image encoder (what it should be)

```
image (BUSI / BreakHis / future)
        |
        v
 preprocessing (resize, normalize, light augment train-only)
        |
        v
 CNN / transfer backbone
   - baseline CNN (from scratch / light)
   - ResNet-50 (ImageNet init, fine-tune)
   - EfficientNet-B0/B3 (candidate)
        |
        v
 global pool → projection MLP → z_img ∈ R^d
        |
        v
 (optional) image-only classifier head
```

**Recommended default for capstone:** **ResNet-50** or **EfficientNet-B0** with ImageNet initialization; keep a **small CNN** as baseline. Pick by validation on BUSI (and later BreakHis), not by brand name.

BUSI has `normal / benign / malignant`. For the **final binary framework**, map:
- malignant → Malignant  
- benign → Benign  
- normal → either excluded from binary head training **or** treated as Benign-with-tag (prefer **exclude from binary fusion head**, keep for optional 3-class image auxiliary task)

---

## 4. Common embedding & fusion strategy

### 4.1 Shared latent size
Choose a common dimension `d` (e.g. **128**). Both branches emit `z_tab, z_img ∈ R^d`.

### 4.2 Fusion options (pick one primary + one ablation)

| Strategy | Idea | Capstone fit |
|---|---|---|
| **A. Late gated fusion (recommended)** | `z = g_tab·z_tab + g_img·z_img` with gates from modality availability | Clear, works with missing modalities |
| **B. Concatenation MLP** | `[z_tab; z_img; m_tab; m_img] → MLP → logits` | Simple, standard |
| **C. Cross-attention fusion** | Attend tabular↔image tokens | Heavier; optional stretch goal |

**Recommended primary:** **B with explicit modality masks** (simplest to justify), ablation with **A**.

```
m_tab, m_img ∈ {0,1}     # 1 if modality present for this forward pass

z_tab_in = z_tab if m_tab else z_tab_missing   # learned missing embedding
z_img_in = z_img if m_img else z_img_missing

h = MLP( concat(z_tab_in, z_img_in, m_tab, m_img) )
P(malignant | inputs) = σ(h)
```

**Critical honesty statement for reports/UI:**  
Public WDBC/WBCD/BUSI are **not paired**. Therefore:

- Training on those sources is **unimodal or missing-modality training**, not “true paired multimodal fusion on the same patient.”
- At inference, if a user supplies **both** tabular + image, the fusion path **can** run as a multimodal forward pass, but we must **not claim** the fusion weights were learned from large paired clinical cohorts unless/until paired data exists.

That is still a valid **framework architecture**; it is just correctly scoped.

---

## 5. Training when samples are unpaired (core strategy)

We use a **multi-task / modality-dropout** schedule so unpaired datasets still teach the encoders.

### Stage I — Unimodal specialization (uses almost all current data)
1. Train **tabular adapters + tabular head** on WDBC and WBCD (schema-separated batches).  
2. Train **image encoder + image head** on BUSI (and later BreakHis).  
3. Keep classical tabular baselines for comparison tables.

### Stage II — Shared latent alignment (still unpaired)
- Freeze or lightly fine-tune encoders.  
- Train projection layers so unimodal heads share the same `d`.  
- Optional: contrastive loss is **not required** for CSE and is risky without pairs—**skip unless paired data appears**.

### Stage III — Fusion head with **synthetic missingness** (not synthetic patients)
On each batch from whichever modality is available:
- Set present modality mask to 1, missing to 0.  
- Use learned `z_*_missing` embeddings.  
- Train fusion head to predict Benign/Malignant from the available side.

This lets **all ~2K (and later 20K) unpaired samples** contribute to learning **without fabricating pairs**.

### Stage IV — Optional true multimodal fine-tune
Only if/when a **paired** dataset exists (same patient tabular+image). Until then, Stage III is the honest fusion training story.

**Leakage controls:**
- Split **by sample_uid / patient_id** within each dataset before any augmentation.  
- Never put the same image/patient in train and test.  
- For BreakHis, prefer **patient-level** splits (known issue in literature).  
- Manifest provenance required for every row.

---

## 6. How the final classifier produces ONE prediction

### Inference API (user-facing)

| User provides | Masks | Path | Output |
|---|---|---|---|
| Tabular only | `m_tab=1, m_img=0` | tabular encoder → fusion with missing image token | Benign/Malignant + SHAP |
| Image only | `m_tab=0, m_img=1` | image encoder → fusion with missing tabular token | Benign/Malignant + Grad-CAM |
| Both | `m_tab=1, m_img=1` | both encoders → fusion | Benign/Malignant + SHAP + Grad-CAM |

```
                 INPUT (any supported subset)
                           |
              +------------+------------+
              |                         |
         TABULAR?                    IMAGE?
              |                         |
              v                         v
      Schema detect/adapter      Preprocess + CNN
              |                         |
              v                         v
           z_tab                      z_img
              |                         |
              +-----------+-------------+
                          |
                   Fusion + masks
                          |
                   Classification head
                          |
                ŷ ∈ {Benign, Malignant}
                p = P(Malignant)
                          |
              +-----------+-----------+
              |                       |
         SHAP on tabular         Grad-CAM on image
         (if tabular given)      (if image given)
```

**Single framework output contract:**
- `label`, `probability`, `modalities_used`, `model_versions`, `explanations{}`, `disclaimer`

Internally many encoders may exist; externally it is **one classification framework**.

---

## 7. Scaling to 20K+ samples

| Phase | Data | Effect on architecture |
|---|---|---|
| Now (~2K) | WDBC + WBCD + BUSI | Prove branches + missing-modality fusion |
| Next (~10K) | + BreakHis (~7.9K images) | Strengthen image encoder; same fusion API |
| Toward 20K+ | + more imaging (e.g. additional histopathology/ultrasound corpora) and/or approved clinical tabular sets with declared schemas | Add adapters; **do not** duplicate rows |

Scaling rule: **more real samples into the correct branch**, same latent width `d`, same fusion head interface.

---

## 8. Where SHAP and Grad-CAM fit

```
Final prediction pathway
        |
        +--> if tabular present:
        |      SHAP on (schema adapter inputs or tabular trunk features)
        |      optional LIME local check
        |
        +--> if image present:
               Grad-CAM / Grad-CAM++ on the CNN feature maps
               (EfficientNet/ResNet last conv block)
```

- SHAP explains **clinical feature contribution** (tabular story).  
- Grad-CAM explains **image regions** (visual story).  
- Fusion does not replace XAI; XAI is attached to the **modalities actually used** for that prediction.

---

## 9. What must change in the current code

| Area | Change |
|---|---|
| `docs/` | Add this architecture as source of truth; update roadmap language from “corpus-only” to “multimodal framework” |
| `src/datasets/` | Keep; extend with `modality`, split manifests, patient keys | 
| New `src/models/tabular_encoder.py` | Schema adapters + shared latent |
| New `src/models/image_encoder.py` | CNN / ResNet / EfficientNet wrappers |
| New `src/models/fusion.py` | Masked fusion + classification head |
| New `src/train_unimodal_*.py` / `train_fusion.py` | Staged training |
| `src/train.py` | Become **tabular baseline / schema-A trainer**, not “the whole system” |
| `src/explain.py` | Split into tabular SHAP + image Grad-CAM modules |
| `streamlit_app.py` / `app.py` | Eventually: tabs for tabular / image / both (later sprint) |
| `configs/` | Add `model.yaml` (d, backbones, masks), keep `datasets.yaml` |

---

## 10. What should remain untouched (for now)

| Keep | Why |
|---|---|
| Existing WDBC load/split/scale path (`src/data.py`) | Stable baseline; still a valid unimodal tabular track |
| Current RF/SVM/… comparison on WDBC (`src/train.py` results) | Literature-facing baseline table |
| Corpus builders (`scripts/build_corpus.py`, validate) | Provenance + 2K milestone evidence |
| Dataset isolation rules | Academic integrity |
| Live Streamlit demo behavior until multimodal UI sprint | Reviewability; don’t break demos mid-design |
| Disclaimer: academic decision-support, not a medical device | Ethics requirement |

---

## 11. Concrete data-flow diagram (training + inference)

### Training (unpaired-capable)

```
Dataset registry (manifest)
        |
        |-- batch type TABULAR (WDBC or WBCD)
        |      adapter_schema → z_tab
        |      m=(1,0) → fusion/unimodal head → loss_cls
        |      (+ SHAP offline for reports)
        |
        |-- batch type IMAGE (BUSI / BreakHis)
               backbone → z_img
               m=(0,1) → fusion/unimodal head → loss_cls
               (+ Grad-CAM offline for reports)
```

### Inference (single final prediction)

```
User input → modality parse
   tabular? → schema adapter → z_tab else z_tab_missing
   image?   → CNN encode   → z_img else z_img_missing
   → fusion(z_tab, z_img, masks) → P(malignant)
   → attach SHAP and/or Grad-CAM for provided modalities
   → return one Benign/Malignant decision package
```

---

## 12. Phased implementation plan (after this design approval)

| Phase | Deliverable | Uses current ~2K how? |
|---|---|---|
| **P0** | This architecture doc + roadmap alignment | Design lock |
| **P1** | Image unimodal trainer (BUSI) + Grad-CAM | 780 images |
| **P2** | Tabular multi-schema adapters (WDBC+WBCD) → shared `z_tab` | 569+675 |
| **P3** | Fusion head + missing-modality masks + unified predict API | All unpaired samples |
| **P4** | UI: tabular / image / both | Demo framework |
| **P5** | BreakHis + scale toward 20K | Image branch growth |
| **P6** | Report: baselines vs fused framework; explicit unpaired caveat | Academic write-up |

---

## 13. Academic claims we will make vs avoid

**Make:**
- Framework supports tabular and image inputs with one decision interface.  
- Encoders learn from large unpaired corpora.  
- Explanations are modality-appropriate (SHAP / Grad-CAM).  
- Corpus scales toward 20K+ without row fabrication.

**Avoid:**
- “Trained multimodal fusion on 20K paired patients” (false with current public sets).  
- “All datasets share one feature table.”  
- “Random Forest is the final multimodal model.”

---

## 14. Decision summary

The corrected architecture is a **missing-modality multimodal framework**:

- **Tabular branch:** schema-specific adapters → shared latent `z_tab`  
- **Image branch:** CNN/ResNet/EfficientNet → shared latent `z_img`  
- **Fusion:** masked combination → **one** Benign/Malignant head  
- **XAI:** SHAP (tabular) + Grad-CAM (image)  
- **Data growth:** real samples into the right branch (BreakHis → 20K path)  
- **Current code:** keep corpus + WDBC baselines; add encoder/fusion layers next—not replace honesty rules

**Next step after your approval:** implement **Phase P1** (BUSI image encoder + Grad-CAM) while keeping the live WDBC demo intact.
