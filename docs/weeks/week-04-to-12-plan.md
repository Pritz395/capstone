# Weeks 4–12 — Planned work

These files are placeholders so the 12-week plan is visible in-repo. Fill each week as you complete it.

---

## Week 4 — Tuning & robustness
- Grid/random search on RF, SVM, XGBoost  
- Stratified k-fold CV scores vs Week 3 holdout  
- Document imbalance handling (class weights if needed)  
- Save: `artifacts/week04_cv_results.csv`

## Week 5 — Deep tabular bake-off
- Tune MLP architecture / early stopping  
- Final tabular model shortlist (top 3)  
- Save: `artifacts/week05_mlp_tuning.json`

## Week 6 — SHAP
- Global importance + beeswarm for best model  
- Local explanations for benign & malignant cases  
- Promote `src/explain.py` from scaffold → week deliverable  
- Save: `artifacts/shap_*` (refresh)

## Week 7 — LIME + narrative
- LIME local HTML for demo cases  
- Map top features → medical meanings in report section  
- Save: short `docs/weeks/week-07.md` write-up

## Week 8 — Web UI polish
- Improve `app.py` UX, error handling, sample cases  
- Show model name, probabilities, top SHAP drivers  
- Deploy locally; screenshot for report

## Week 9 — Extension (pick one)
- **A:** Image model + Grad-CAM on BreakHis/BUSI, *or*  
- **B:** Extra evaluation (threshold analysis, calibration, bias notes)  
- Must not regress Week 3–8 tabular path

## Week 10 — Evaluation freeze
- Lock final metrics table  
- Ablation: with/without scaling, top-k features  
- Limitations + research gap finalized

## Week 11 — Report & docs
- Capstone report (intro → lit → method → results → XAI → conclusion)  
- README polished for external readers  
- Ethics / “not a medical device” section

## Week 12 — Demo & handoff
- Presentation slides  
- Live or recorded demo  
- Tag release `v1.0-week12` on GitHub  
