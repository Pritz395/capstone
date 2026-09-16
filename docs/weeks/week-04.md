# Week 4 — Modality-aware architecture foundation

**Status:** Complete (foundation only — no final multimodal model)

## Story

We prepared the repository so Weeks 5–12 can add CNNs, fusion, and XAI **without rewriting** the Week 3 tabular baselines.

## What was added

Modality boundaries:

```
Dataset Registry
       │
       ├── Tabular datasets (WDBC, WBCD)
       │      → src/preprocessing/tabular.py
       │      → src/models/tabular.py
       │      → src/train_tabular.py
       │
       └── Image datasets (BUSI now; BreakHis later)
              → src/preprocessing/image.py   (skeleton)
              → src/models/image.py          (encoder interface skeleton)
```

Also retained: `docs/ARCHITECTURE.md` describing the **future** encoder → latent → fusion design.

## What this enables later

- BreakHis / more image sets
- CNN / ResNet-50 / EfficientNet training
- Missing-modality multimodal fusion
- SHAP (tabular) + Grad-CAM (images)
- UI accepting tabular and/or image

## What we intentionally did NOT build in Week 4

- No CNN / ResNet / EfficientNet training  
- No multimodal fusion  
- No fabricated paired tabular+image records  
- No Grad-CAM / final XAI stack  
- No claim that fusion already exists  

## Current architecture (Week 4)

```
DATA CORPUS (2,024)
        │
   ┌────┴────┐
TABULAR    IMAGE
WDBC+WBCD  BUSI(+future)
   │          │
   ▼          ▼
tabular     image registry
baselines   + preprocess/encoder SKELETONS
   │
   ▼
model comparison (per schema)
```

## Next (Week 5+)

Image experiments on BUSI → later fusion + 20K expansion + unified prediction UI.
