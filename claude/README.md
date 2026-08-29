# Seoul Bike Demand — Streamlit App

Predicts hourly bike rental demand in Seoul using a tuned Gradient Boosting
Regressor trained on the SeoulBikeData dataset.

## Files in this folder

- `train_model.py` — reproduces the notebook's feature engineering + trains
  the best model (Gradient Boosting), saves it to `model.pkl`
- `app.py` — the Streamlit app that loads `model.pkl` and serves predictions
- `requirements.txt` — dependencies for Streamlit Cloud (unpinned versions,
  so pip picks whatever is compatible with the Python version Streamlit
  Cloud runs)
- `model.pkl` — the trained pipeline (YOU generate this, see step 1 below —
  it is not included here, you must create and commit it yourself)

## Step 1 — Train the model locally

You need `SeoulBikeData.csv` (the raw dataset) in this same folder.

```bash
pip install -r requirements.txt
python train_model.py
```

This prints test MAE/RMSE/R² and creates `model.pkl` right here in the
folder. Grid search can take a couple of minutes — that's expected.

## Step 2 — Test the app locally (recommended)

```bash
streamlit run app.py
```

Open the local URL it prints (usually http://localhost:8501) and confirm
predictions work before deploying.

## Step 3 — Push everything to GitHub

From this folder:

```bash
git init
git add app.py train_model.py requirements.txt model.pkl README.md
git commit -m "Seoul bike demand streamlit app"
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repo>.git
git push -u origin main
```

**Critical:** `model.pkl` MUST be committed and pushed. Streamlit Cloud
only runs `app.py` — it never runs `train_model.py` for you. If `app.py`
can't find `model.pkl` in the repo, the app crashes on startup.

If you put these files inside a subfolder of your repo (e.g.
`myproject/app.py` instead of `app.py` at the repo root), remember that
subfolder path — you'll need it in Step 4.

Do **not** commit `SeoulBikeData.csv` unless you're sure you can
redistribute it — the deployed app only needs `model.pkl`, not the raw data.

## Step 4 — Deploy on Streamlit Community Cloud

1. Go to https://share.streamlit.io and sign in with GitHub.
2. Click "New app".
3. Pick your repo and branch (`main`).
4. Set **"Main file path"** correctly:
   - If your files sit at the repo root → `app.py`
   - If they sit in a subfolder (e.g. `myproject/`) → `myproject/app.py`
   Getting this wrong is a very common cause of deploy failures.
5. Click "Deploy". Streamlit Cloud installs `requirements.txt` and runs
   your app automatically.
6. You'll get a public URL like:
   `https://<your-app-name>-<random-id>.streamlit.app`

## Common deploy errors and fixes

- **"Error installing requirements"** — usually one of:
  - `model.pkl` is missing from the repo (app crashes, Streamlit shows a
    generic error screen)
  - Pinned package versions in `requirements.txt` don't support the Python
    version Streamlit Cloud is using — use unpinned versions like this
    repo does
- **`ERROR: Could not open requirements file: ... 'requirements.txt,'`**
  (note the trailing comma) — this is a typo in the Streamlit Cloud app's
  Settings, not your code. Go to Manage app → Settings and check the
  "Main file path" / any custom requirements path field for a stray comma
  or extra text, remove it, save, and reboot.
- **App loads but "Main file path" 404s** — the path doesn't match where
  `app.py` actually lives in your repo. Fix it in Manage app → Settings →
  General.

## Updating the app later

Any `git push` to `main` auto-redeploys the app on Streamlit Cloud. If you
retrain the model, just re-run `train_model.py`, commit the new
`model.pkl`, and push.
