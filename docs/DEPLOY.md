# Deploy the demo UI (always-on)

The app is deploy-ready on GitHub (`render.yaml`, `Dockerfile`, `Procfile`, inference models committed).

## Option A — Render (recommended, free)

1. Open this link (signed in with the GitHub account that owns the repo):  
   **https://render.com/deploy?repo=https://github.com/Pritz395/capstone**
2. Click **Apply** / create the Blueprint service `capstone-breast-xai`
3. Wait for the first build (a few minutes)
4. Open the `.onrender.com` URL Render gives you

Cold starts on the free plan can take ~30–60s after idle.

## Option B — Local + Cloudflare tunnel (temporary public URL)

Keep the laptop awake and Flask running:

```bash
cd /Users/preetham/Desktop/Projekt/capstone
source .venv/bin/activate
python app.py
# other terminal:
cloudflared tunnel --url http://127.0.0.1:5000
```

Use the `https://….trycloudflare.com` link it prints. This dies when you stop the tunnel or sleep the machine.

## What’s already in the repo for hosting

| File | Purpose |
|---|---|
| `requirements-deploy.txt` | Lean prod deps (Flask, sklearn, SHAP, gunicorn) |
| `Procfile` / `render.yaml` | Render start command |
| `Dockerfile` | Alternate container deploy |
| `models/best_model.joblib` (+ scaler, feature_cols) | Shipped so cloud doesn’t need to retrain |
| `/health` | Health check endpoint |
