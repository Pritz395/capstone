# Professor demo — what to show tomorrow

## Start the UI (before the meeting)

```bash
cd /Users/preetham/Desktop/Projekt/capstone
source .venv/bin/activate
python app.py
```

Open: **http://127.0.0.1:5000**

If models are missing:

```bash
python -m src.train
python -m src.explain
python app.py
```

---

## 60-second script (“show me the model”)

1. **One sentence**  
   “We’ve built a classifier on the Wisconsin breast-cancer dataset that predicts benign vs malignant and explains the decision with SHAP.”

2. **Show the UI**  
   - Click **Load malignant sample** → it predicts and shows probability  
   - Point at **Why this prediction?** (top features)  
   - Click **Load benign sample** → contrast the result  

3. **Scroll to model comparison**  
   - “We compared 9 models; Random Forest is deployed at ~**97.4%** test accuracy, recall ~93%, ROC-AUC ~0.99.”

4. **If they ask for code**  
   - `src/train.py` — training + metrics  
   - `models/best_model.joblib` — saved model  
   - `artifacts/leaderboard.csv` — numbers  

5. **If they ask “is this medical?”**  
   - “Academic decision-support prototype only — not a clinical device.”

---

## Optional extras (if they dig deeper)

| Ask | Show |
|---|---|
| Data? | `data/wdbc/` + Week 2 EDA plots in `artifacts/eda/` |
| Plan? | `docs/ROADMAP_12_WEEKS.md` (we’re at Week 3) |
| Papers? | `papers/` + `research/00_RESOURCE_INVENTORY.md` |
| GitHub? | https://github.com/Pritz395/capstone |

---

## Do **not** claim

- That it replaces doctors  
- That image/CNN/Grad-CAM is fully built yet (that’s later / optional)  
- Perfect clinical validation — this is a CSE Week‑3 milestone  
