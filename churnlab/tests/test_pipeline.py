import numpy as np
import polars as pl
from churnlab.data import build_raw
from churnlab.features import build_features
from churnlab.drift import psi, drift_report


def test_features_one_row_per_user():
    users, events, labels = build_raw(500)
    f = build_features(users, events)
    assert f.height == users.height and f["user_id"].n_unique() == f.height
    assert f["ev_total"].min() >= 0


def test_psi_identical_is_zero_and_shift_detected():
    x = np.random.default_rng(0).normal(size=5000)
    assert psi(x, x) < 1e-6
    assert psi(x, x + 2) > 0.25


def test_drift_report_flags_shift():
    a = pl.DataFrame({"v": np.random.default_rng(1).normal(size=3000)})
    b = pl.DataFrame({"v": np.random.default_rng(2).normal(3, 1, size=3000)})
    assert drift_report(a, b, ["v"])["status"][0] == "ALERT"
