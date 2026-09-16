# Deploy the demo UI (always-on, no paid plan)

## Live URL

**https://capstone-jjgpitxnzxxtmljc6fhs75.streamlit.app/**

Hosted on Streamlit Community Cloud (free). Render was skipped (payment wall).

## Option A — Streamlit Community Cloud (done)

1. Open **https://share.streamlit.io/** (or https://streamlit.io/cloud)
2. Sign in with **GitHub** (`Pritz395`)
3. **Create app** → pick repo `Pritz395/capstone` → branch `main`
4. Main file: **`streamlit_app.py`**
5. Deploy → permanent `*.streamlit.app` URL

To redeploy after code changes: push to `main` (Streamlit usually auto-updates) or click **Reboot** / **Rerun** in the app settings.

## Option B — Temporary public URL (laptop must stay awake)

```bash
# from repo root
source .venv/bin/activate
python app.py
# other terminal:
cloudflared tunnel --url http://127.0.0.1:5000
```

## Local Streamlit (optional)

```bash
source .venv/bin/activate
pip install -r requirements-dev.txt   # full ML stack for training
streamlit run streamlit_app.py
```

## Repo hosting notes

| File | Purpose |
|---|---|
| `streamlit_app.py` | Free-cloud demo UI |
| `requirements.txt` | Lean deps for Streamlit Cloud |
| `requirements-dev.txt` | Full local stack (XGBoost, CatBoost, etc.) |
| `models/best_model.joblib` (+ scaler, feature_cols) | Shipped so cloud doesn’t retrain |
| `app.py` | Original Flask UI (local / optional) |
