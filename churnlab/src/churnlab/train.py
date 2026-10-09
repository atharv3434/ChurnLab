"""Train: Optuna-tuned LightGBM, isotonic calibration, SHAP explainability, saved artifacts."""
from __future__ import annotations
import json, pickle
from pathlib import Path
import numpy as np
import pandas as pd
import polars as pl
import lightgbm as lgb
import optuna
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

from .data import build_raw
from .features import build_features, CATEGORICAL
from .drift import drift_report

ART = Path(__file__).resolve().parents[2] / "artifacts"
optuna.logging.set_verbosity(optuna.logging.WARNING)


def to_X(df: pl.DataFrame, feats: list[str], cats: dict[str, list[str]]) -> pd.DataFrame:
    X = df.select(feats).to_pandas()
    for c in CATEGORICAL:
        X[c] = pd.Categorical(X[c].astype(str), categories=cats[c])
    return X


def shap_2d(sv) -> np.ndarray:
    sv = sv[1] if isinstance(sv, list) else sv
    sv = np.asarray(sv)
    return sv[..., 1] if sv.ndim == 3 else sv


def run(n_users: int = 20_000, n_trials: int = 20, seed: int = 42) -> dict:
    ART.mkdir(exist_ok=True)
    users, events, labels = build_raw(n_users, seed)
    df = build_features(users, events).join(labels, on="user_id")
    feats = [c for c in df.columns if c not in ("user_id", "churned")]
    numeric = [c for c in feats if c not in CATEGORICAL]

    idx = np.random.default_rng(seed).permutation(len(df))
    a, b = int(.6 * len(df)), int(.8 * len(df))
    tr, ca, te = df[idx[:a]], df[idx[a:b]], df[idx[b:]]
    cats = {c: sorted(df[c].unique().to_list()) for c in CATEGORICAL}
    Xtr, Xca, Xte = (to_X(d, feats, cats) for d in (tr, ca, te))
    ytr, yca, yte = tr["churned"].to_numpy(), ca["churned"].to_numpy(), te["churned"].to_numpy()

    def fit(params):
        m = lgb.LGBMClassifier(objective="binary", verbosity=-1, n_estimators=400,
                               subsample_freq=1, random_state=seed, **params)
        m.fit(Xtr, ytr, eval_set=[(Xca, yca)], eval_metric="auc",
              callbacks=[lgb.early_stopping(30, verbose=False)])
        return m

    def objective(t: optuna.Trial) -> float:
        m = fit(dict(
            learning_rate=t.suggest_float("learning_rate", 0.02, 0.2, log=True),
            num_leaves=t.suggest_int("num_leaves", 8, 64),
            min_child_samples=t.suggest_int("min_child_samples", 10, 100),
            subsample=t.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=t.suggest_float("colsample_bytree", 0.5, 1.0),
            reg_lambda=t.suggest_float("reg_lambda", 1e-3, 10, log=True)))
        return roc_auc_score(yca, m.predict_proba(Xca)[:, 1])

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials)
    model = fit(study.best_params)

    iso = IsotonicRegression(out_of_bounds="clip").fit(model.predict_proba(Xca)[:, 1], yca)
    raw = model.predict_proba(Xte)[:, 1]
    cal = iso.predict(raw)
    baseline = lgb.LGBMClassifier  # noqa (placeholder to keep imports tidy)
    metrics = {
        "roc_auc": round(roc_auc_score(yte, cal), 4),
        "pr_auc": round(average_precision_score(yte, cal), 4),
        "brier_raw": round(brier_score_loss(yte, raw), 4),
        "brier_calibrated": round(brier_score_loss(yte, cal), 4),
        "base_rate": round(float(yte.mean()), 4),
        "best_params": study.best_params, "n_trials": n_trials,
    }

    sample = Xte.sample(min(1500, len(Xte)), random_state=seed)
    sv = shap_2d(shap.TreeExplainer(model).shap_values(sample))
    imp = dict(sorted(zip(feats, np.abs(sv).mean(0).round(4).tolist()), key=lambda kv: -kv[1]))
    metrics["shap_importance"] = imp
    plt.figure(figsize=(7, 4.5))
    plt.barh(list(imp)[::-1], list(imp.values())[::-1], color="#4f46e5")
    plt.xlabel("mean |SHAP value|"); plt.title("What drives churn predictions")
    plt.tight_layout(); plt.savefig(ART / "shap_importance.png", dpi=140); plt.close()

    shifted = te.with_columns((pl.col("ev_30d") * 0.5).alias("ev_30d"),
                              (pl.col("failed_payments") + 1).alias("failed_payments"))
    metrics["drift_demo"] = drift_report(tr, shifted, numeric).head(5).to_dicts()

    with open(ART / "model.pkl", "wb") as f:
        pickle.dump({"model": model, "iso": iso, "features": feats, "cats": cats}, f)
    (ART / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    m = run()
    print(json.dumps({k: v for k, v in m.items() if k not in ("shap_importance",)}, indent=2))
    print("Top SHAP:", list(m["shap_importance"].items())[:5])
