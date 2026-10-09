# ChurnLab: Customer Churn Prediction, End to End

A compact, production-shaped data science project on a modern Python stack.

| Stage | Tech | What it does |
|---|---|---|
| Data generation | **Polars** + NumPy | Synthetic users + ~800k-row event log with a hidden engagement signal |
| Feature engineering | **DuckDB** SQL over Arrow (zero-copy from Polars) | Rolling windows, usage momentum, recency, ratios |
| Modelling | **LightGBM** (native categoricals) | Gradient-boosted classifier |
| Tuning | **Optuna** (TPE sampler) | 20-trial hyperparameter search with early stopping |
| Calibration | Isotonic regression | Probabilities usable for business decisions |
| Explainability | **SHAP** | Global importance plot + per-prediction reasons |
| Monitoring | PSI drift detector | Flags feature distribution shift (ALERT > 0.25) |
| Serving | **FastAPI** + Pydantic v2 | `/predict` returns probability, risk tier, top-3 reasons |
| Quality | pytest | Feature, drift and API tests |

## Run it
```bash
pip install -r requirements.txt
export PYTHONPATH=src
make train     # builds data, tunes, trains, writes artifacts/
make test
make serve     # http://127.0.0.1:8000/docs
```

Example request:
```bash
curl -X POST localhost:8000/predict -H 'content-type: application/json' -d '{
 "plan":"free","country":"IN","acq_channel":"ads","tenure_days":120,"ev_total":8,
 "ev_7d":0,"ev_30d":1,"ev_prev_30d":5,"tickets":3,"failed_payments":2,
 "feature_uses":2,"days_since_last_event":40}'
```

## Results (synthetic data, seed 42)
ROC-AUC 0.676, PR-AUC 0.456 (base rate 0.319). The labels are deliberately noisy,
so this is a realistic ceiling rather than a leaderboard number. The raw model was
already well calibrated (Brier 0.1995 vs 0.2010 after isotonic), so calibration is
a safeguard here and not a big win. The SHAP plot shows `ev_total` and `plan` as top drivers,
matching how the data was generated.

## Ideas to extend
- Swap synthetic data for a real dataset (e.g. Telco churn or KKBox) and re-use `features.py`
- Add MLflow or Weights & Biases tracking around `train.run`
- Add a Streamlit dashboard reading `artifacts/metrics.json`
- Time-based split + backtesting instead of random split
- Dockerfile + CI running `make test`
