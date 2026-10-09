"""Population Stability Index (PSI) drift monitor for numeric features."""
from __future__ import annotations
import numpy as np
import polars as pl


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, edges)[0] / len(expected)
    a = np.histogram(actual, edges)[0] / len(actual)
    e, a = np.clip(e, 1e-6, None), np.clip(a, 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))


def drift_report(reference: pl.DataFrame, current: pl.DataFrame, cols: list[str]) -> pl.DataFrame:
    rows = []
    for c in cols:
        v = psi(reference[c].drop_nulls().to_numpy(), current[c].drop_nulls().to_numpy())
        rows.append({"feature": c, "psi": round(v, 4),
                     "status": "ALERT" if v > 0.25 else "WARN" if v > 0.1 else "ok"})
    return pl.DataFrame(rows).sort("psi", descending=True)
