# Datasets & 2K+ corpus strategy

## Assessment (current repo)

| Item | Status |
|---|---|
| WDBC schema | 30 continuous FNA nuclear morphometry features + binary B/M |
| Features used by models | Exactly those 30 columns (`src/data.py` `FEATURE_NAMES`) |
| Extra local copies | `wdbc.data`, `breast-cancer.csv`, Kaggle `data.csv` — **same WDBC**, not new samples |
| Compatible giant twin of WDBC | **None publicly available** with the same 30-feature definition |

**You cannot honestly row-bind random breast-cancer CSVs into WDBC** just because names look similar. Different instruments, encodings, and semantics = leakage / invalid science.

---

## Recommended 2K milestone (technically defensible)

Build a **multi-source corpus with provenance**, where each dataset keeps its own `schema_id`. Count **valid samples across the framework**, but **train models only within a schema**.

| Dataset | Source | ~N | Features | Compatible with WDBC columns? | Role |
|---|---|---:|---|---|---|
| **WDBC** | UCI / local `data/wdbc` | 569 | 30 continuous morphometry | — (reference) | Existing ML track |
| **WBCD Original** | [UCI id=15](https://archive.ics.uci.edu/dataset/15/breast+cancer+wisconsin+original) | 699 (≤16 missing bare nuclei dropped) | 9 ordinal cytology scores | **NO** | Second tabular track |
| **BUSI** | Al-Dhabyani et al. / [HF MedOtter/BUSI](https://huggingface.co/datasets/MedOtter/BUSI) | 780 images | Ultrasound PNG + masks | **NO** (image modality) | Image track (CNN later) |
| **BreakHis** (later) | [UFPR](http://www.inf.ufpr.br/vri/databases/BreaKHis_v1.tar.gz) | ~7,909 images | Histopathology | **NO** | Path toward 20K |

**Immediate verified total (this repo):** **2,024** valid samples  
`WDBC 569 + WBCD 675 (after dropping missing/`?` bare nuclei + exact dupes) + BUSI 780`.

Licensing notes:
- WDBC / WBCD: UCI citation required (Wolberg et al.)
- BUSI: cite Al-Dhabyani et al., Data in Brief 2020; CC/research use per source page
- BreakHis: non-commercial research; cite Spanhol et al., IEEE TBME 2016

---

## What we explicitly do NOT do

- Merge WDBC + WBCD into one feature matrix
- Oversample / duplicate rows to hit 2,000
- Impute fabricated labels for missing BUSI/WBCD fields
- Pretend ultrasound pixels are WDBC `radius_mean`

---

## Pipeline

```
configs/datasets.yaml
        │
        ▼
src/datasets/* loaders ──► LoadedDataset(manifest, optional X/y)
        │
        ▼
scripts/build_corpus.py ──► artifacts/corpus/manifest.csv
                         ──► artifacts/corpus/summary.csv
        │
        ▼
scripts/validate_corpus.py
        │
        ├── schema wdbc_v1_30features  → existing src/train.py (unchanged default)
        ├── schema wbcd_v1_9features   → separate tabular track (future/optional)
        └── schema busi_v1_image       → catalog now; CNN in later week
```

Every row in `manifest.csv` has: `sample_uid`, `dataset_id`, `schema_id`, `modality`, `label_raw`, `label_binary`, `source_path`.

---

## How to run (verify ≥2000)

```bash
# from repo root
source .venv/bin/activate
pip install -r requirements-dev.txt

# Fetch open datasets (WBCD + BUSI). WDBC is already local.
python scripts/fetch_datasets.py --all

# Build corpus + enforce mentor minimum
python scripts/build_corpus.py --min-samples 2000

# Integrity checks (dedupe UIDs, schema isolation)
python scripts/validate_corpus.py

# Existing WDBC model training still works
python -m src.train
```

Check:

```bash
cat artifacts/corpus/summary.csv
# ALL row n_samples should be >= 2000
```

---

## How to add another dataset later (toward 20K)

1. Create `src/datasets/<name>.py` with `load(spec, root) -> LoadedDataset`
2. Register loader in `src/datasets/registry.py` `LOADERS`
3. Add an entry under `datasets:` in `configs/datasets.yaml`
4. Place files under `data/<name>/` (do not commit huge binaries)
5. Run `python scripts/build_corpus.py --min-samples 2000`
6. Only call `schema_feature_matrix(..., schema_id=...)` when features are **truly** the same schema

Suggested next add: **BreakHis** (`enabled: true` after download) → corpus jumps by ~7.9K images.
