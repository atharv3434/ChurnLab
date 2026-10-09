"""Feature engineering in DuckDB SQL over Polars frames (zero-copy via Arrow)."""
from __future__ import annotations
import duckdb
import polars as pl

FEATURE_SQL = """
WITH ev AS (
    SELECT
        user_id,
        COUNT(*)                                                          AS ev_total,
        COUNT(*) FILTER (WHERE ts >= TIMESTAMP '2025-01-01' - INTERVAL 7 DAY)  AS ev_7d,
        COUNT(*) FILTER (WHERE ts >= TIMESTAMP '2025-01-01' - INTERVAL 30 DAY) AS ev_30d,
        COUNT(*) FILTER (WHERE ts <  TIMESTAMP '2025-01-01' - INTERVAL 30 DAY
                           AND ts >= TIMESTAMP '2025-01-01' - INTERVAL 60 DAY) AS ev_prev_30d,
        COUNT(*) FILTER (WHERE event = 'support_ticket')                  AS tickets,
        COUNT(*) FILTER (WHERE event = 'payment_failed')                  AS failed_payments,
        COUNT(*) FILTER (WHERE event = 'feature_use')                     AS feature_uses,
        DATE_DIFF('day', MAX(ts), TIMESTAMP '2025-01-01')                 AS days_since_last_event
    FROM events GROUP BY user_id
)
SELECT
    u.user_id, u.plan, u.country, u.acq_channel,
    DATE_DIFF('day', u.signup_date, TIMESTAMP '2025-01-01') AS tenure_days,
    COALESCE(e.ev_total, 0)        AS ev_total,
    COALESCE(e.ev_7d, 0)           AS ev_7d,
    COALESCE(e.ev_30d, 0)          AS ev_30d,
    COALESCE(e.ev_prev_30d, 0)     AS ev_prev_30d,
    COALESCE(e.tickets, 0)         AS tickets,
    COALESCE(e.failed_payments, 0) AS failed_payments,
    COALESCE(e.feature_uses, 0)    AS feature_uses,
    COALESCE(e.days_since_last_event, 999) AS days_since_last_event,
    -- momentum: >1 means usage is accelerating, <1 decaying
    (COALESCE(e.ev_30d, 0) + 1.0) / (COALESCE(e.ev_prev_30d, 0) + 1.0) AS usage_momentum,
    COALESCE(e.feature_uses, 0) * 1.0 / NULLIF(e.ev_total, 0)          AS feature_ratio
FROM users u LEFT JOIN ev e USING (user_id)
ORDER BY u.user_id
"""

CATEGORICAL = ["plan", "country", "acq_channel"]


def build_features(users: pl.DataFrame, events: pl.DataFrame) -> pl.DataFrame:
    con = duckdb.connect()
    con.register("users", users.to_arrow())
    con.register("events", events.to_arrow())
    return con.execute(FEATURE_SQL).pl()
