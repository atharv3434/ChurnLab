from fastapi.testclient import TestClient
from churnlab.api import app

SAMPLE = dict(plan="free", country="IN", acq_channel="ads", tenure_days=120, ev_total=8, ev_7d=0,
              ev_30d=1, ev_prev_30d=5, tickets=3, failed_payments=2, feature_uses=2, days_since_last_event=40)


def test_predict_roundtrip():
    with TestClient(app) as c:
        assert c.get("/health").json()["status"] == "ok"
        r = c.post("/predict", json=SAMPLE).json()
        assert 0 <= r["churn_probability"] <= 1 and len(r["top_reasons"]) == 3
        assert c.post("/predict", json={**SAMPLE, "plan": "gold"}).status_code == 422
