# NEPSE ML Ensemble Pipeline

ML ensemble (XGBoost + LightGBM + CatBoost/Logistic) trading-probability pipeline
for NEPSE-listed equities. Companion project to the main rule-based research app
(`New-Chataut_Nepse-Analysis`) — this repo is for model research/training, run
from Google Colab, not deployed as an app.

Data source (same one the main app's daily sync uses):
`github.com/Aabishkar2/nepse-data` — per-company CSVs under `data/company-wise/`.

## Modules
- [x] **Module 1 — Data Preprocessing** (`src/module1_preprocessing.py`): corporate-action
  back-adjustment, rolling-z-score outlier filtering, illiquid-ticker drop.
  Verified on the full 372-symbol universe: 321 kept, 314 corp-action adjustments,
  219 bad ticks cleaned.
- [ ] Module 2 — Feature engineering + target labelling
- [ ] Module 3 — Base model training (XGBoost / LightGBM / logistic baseline)
- [ ] Module 4 — Ensembling, calibration, walk-forward validation
- [ ] Module 5 — Out-of-time backtest

## Colab workflow
1. Push this repo to GitHub.
2. Open `notebooks/01_preprocessing.ipynb` in Colab (File > Open notebook > GitHub, paste repo URL).
3. Run the cells top to bottom — it clones the repo, pulls fresh NEPSE data, installs
   `requirements.txt`, and runs Module 1.
4. Commit/push any changes back to GitHub from Colab (or download + push manually)
   so history stays versioned like the main project.

## Repo structure
```
src/                      pipeline modules (plain .py, importable + runnable standalone)
notebooks/                Colab notebooks, one per module (or combined)
data/raw/                 gitignored — pulled fresh each Colab run
data/processed/           gitignored — regenerated each Colab run
requirements.txt
```
