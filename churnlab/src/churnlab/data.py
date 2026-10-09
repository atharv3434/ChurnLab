"""
Synthetic subscription-business data (users + event log) generated with Polars.

Churn is driven by *behavioural* signals hidden in the event log (declining usage,
support tickets, failed payments), so feature engineering genuinely matters.
"""
from __future__ import annotations
import numpy as np
import polars as pl

SNAPSHOT = np.datetime64("2025-01-01")
KINDS = np.array(["login", "feature_use", "support_ticket", "payment_failed"])


def make_users(n: int = 20_000, seed: int = 42) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    signup_offset = rng.integers(30, 720, n)
    return pl.DataFrame({
        "user_id": np.arange(n),
        "signup_date": (SNAPSHOT - signup_offset.astype("timedelta64[D]")).astype("datetime64[us]"),
        "plan": rng.choice(["free", "basic", "pro", "team"], n, p=[.35, .30, .25, .10]),
        "country": rng.choice(["IN", "US", "DE", "BR", "UK", "JP"], n, p=[.25, .25, .12, .13, .13, .12]),
        "acq_channel": rng.choice(["organic", "ads", "referral", "partner"], n, p=[.4, .3, .2, .1]),
        "_engagement": rng.beta(2, 2.5, n),  # latent driver of both events and churn
    })


def make_events(users: pl.DataFrame, seed: int = 7) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    eng = users["_engagement"].to_numpy()
    uid = users["user_id"].to_numpy()
    signup = users["signup_date"].to_numpy().astype("datetime64[D]")
    ages = (SNAPSHOT - signup).astype(int)
    counts = rng.poisson(5 + 60 * eng)
    decay = rng.uniform(0, 1, len(uid)) * (1 - eng)  # at-risk users go quiet recently
    total = counts.sum()
    u = np.repeat(uid, counts)
    scale = np.repeat(25 * (1 - decay) + 5, counts)
    age_rep = np.repeat(ages, counts)
    days_ago = np.minimum(rng.exponential(scale), age_rep).astype(int)
    p_bad = np.repeat(0.02 + 0.10 * (1 - eng), counts)
    r = rng.random(total)
    kind = np.where(r < p_bad, 2, np.where(r < p_bad * 1.6, 3, np.where(r < p_bad * 1.6 + 0.38, 1, 0)))
    ts = (SNAPSHOT - days_ago.astype("timedelta64[D]")).astype("datetime64[us]")
    return pl.DataFrame({"user_id": u, "ts": ts, "event": KINDS[kind]})


def make_labels(users: pl.DataFrame, seed: int = 99) -> pl.DataFrame:
    """Churn within 30 days after snapshot; depends on latent engagement + plan."""
    rng = np.random.default_rng(seed)
    plan_risk = users["plan"].replace_strict({"free": .8, "basic": .3, "pro": -.2, "team": -.6}).to_numpy()
    logit = -1.0 + 3.2 * (0.45 - users["_engagement"].to_numpy()) + plan_risk * 0.6
    p = 1 / (1 + np.exp(-logit))
    return pl.DataFrame({"user_id": users["user_id"], "churned": (rng.random(len(p)) < p).astype(int)})


def build_raw(n: int = 20_000, seed: int = 42):
    users = make_users(n, seed)
    events = make_events(users)
    labels = make_labels(users)
    return users.drop("_engagement"), events, labels
