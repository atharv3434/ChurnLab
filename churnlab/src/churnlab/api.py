"""FastAPI service: calibrated churn probability + per-request SHAP reasons."""
from __future__ import annotations
import pickle
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal
import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI
from pydantic import BaseModel, Field

ART = Path(__file__).resolve().parents[2] / "artifacts"
state: dict = {}


class Customer(BaseModel):
    plan: Literal["free", "basic", "pro", "team"]
    country: Literal["IN", "US", "DE", "BR", "UK", "JP"]
    acq_channel: Literal["organic", "ads", "referral", "partner"]
    tenure_days: int = Field(ge=0)
    ev_total: int = Field(ge=0)
    ev_7d: int = Field(ge=0)
    ev_30d: int = Field(ge=0)
    ev_prev_30d: int = Field(ge=0)
    tickets: int = Field(ge=0)
    failed_payments: int = Field(ge=0)
    feature_uses: int = Field(ge=0)
    days_since_last_event: int = Field(ge=0)

    def to_frame(self, feats, cats) -> pd.DataFrame:
        d = self.model_dump()
        d["usage_momentum"] = (d["ev_30d"] + 1.0) / (d["ev_prev_30d"] + 1.0)
        d["feature_ratio"] = d["feature_uses"] / d["ev_total"] if d["ev_total"] else np.nan
        X = pd.DataFrame([d])[feats]
        for c, levels in cats.items():
            X[c] = pd.Categorical(X[c], categories=levels)
        return X


@asynccontextmanager
async def lifespan(_: FastAPI):
    p = ART / "model.pkl"
    if not p.exists():
        raise RuntimeError("Run `python -m churnlab.train` first.")
    with open(p, "rb") as f:
        bundle = pickle.load(f)
    state.update(bundle, explainer=shap.TreeExplainer(bundle["model"]))
    yield
    state.clear()


app = FastAPI(title="ChurnLab API", version="1.0", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok", "n_features": len(state["features"])}


@app.post("/predict")
def predict(c: Customer, top_k: int = 3):
    X = c.to_frame(state["features"], state["cats"])
    raw = float(state["model"].predict_proba(X)[0, 1])
    prob = float(state["iso"].predict([raw])[0])
    sv = state["explainer"].shap_values(X)
    sv = sv[1] if isinstance(sv, list) else sv
    sv = np.asarray(sv)
    sv = sv[0, :, 1] if sv.ndim == 3 else sv[0]
    reasons = []
    for i in np.argsort(-np.abs(sv))[:top_k]:
        v = X.iloc[0, i]
        reasons.append({"feature": state["features"][i],
                        "value": v.item() if hasattr(v, "item") else str(v),
                        "effect": "raises risk" if sv[i] > 0 else "lowers risk",
                        "shap": round(float(sv[i]), 3)})
    tier = "high" if prob >= .5 else "medium" if prob >= .25 else "low"
    return {"churn_probability": round(prob, 4), "risk_tier": tier, "top_reasons": reasons}
