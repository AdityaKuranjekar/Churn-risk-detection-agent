import pytest
import os
import sys
import numpy as np
from types import SimpleNamespace
import sqlite3
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from app.services.ml_model import (
        _MODEL, _COLS, _PRE, _CARD,
        build_feature_dict, predict, predict_many, band, explain, predict_from_customer, health,
        ModelNotLoaded
    )
    from ml.features import BASE_COLS
    HAS_MODEL = True
except ModelNotLoaded:
    HAS_MODEL = False

@pytest.fixture(autouse=True)
def skip_if_no_model():
    if not HAS_MODEL:
        pytest.skip("Model artifacts missing. Run ml/train_model.py first.")

# Exact dicts from Step 2 self_check
healthy_row = {
    "tenure_days": 700, "monthly_charges": 20.0, "arr": 240.0,
    "plan_tier_ord": 0, "num_devices": 1, "age": 40,
    "payment_failures": 0, "support_contacts": 0, "engagement_score": 90,
    "last_login_days": 1, "usage_level": 50.0
}

mid_row = {
    "tenure_days": 180, "monthly_charges": 60.0, "arr": 720.0,
    "plan_tier_ord": 1, "num_devices": 2, "age": 35,
    "payment_failures": 1, "support_contacts": 1, "engagement_score": 40,
    "last_login_days": 7, "usage_level": 15.0
}

critical_row = {
    "tenure_days": 30, "monthly_charges": 120.0, "arr": 1440.0,
    "plan_tier_ord": 2, "num_devices": 4, "age": 25,
    "payment_failures": 3, "support_contacts": 5, "engagement_score": 5,
    "last_login_days": 21, "usage_level": 2.0
}


def test_model_loaded():
    assert _MODEL is not None
    assert _COLS is not None
    assert len(_COLS) == _MODEL.n_features_in_

def test_predict_monotonic():
    p_healthy = predict(healthy_row)
    p_mid = predict(mid_row)
    p_critical = predict(critical_row)
    
    assert p_healthy < 0.15
    assert p_healthy < p_mid < p_critical
    assert p_critical > 0.80

def test_predict_missing_keys():
    # Drop some keys
    incomplete_row = healthy_row.copy()
    del incomplete_row["age"]
    del incomplete_row["engagement_score"]
    
    p = predict(incomplete_row)
    assert 0.0 <= p <= 1.0

def test_predict_empty():
    p = predict({})
    assert 0.0 <= p <= 1.0

def test_predict_absurd_inputs():
    absurd_row = {
        "tenure_days": -5,
        "payment_failures": 999,
        "usage_level": 1e9,
        "monthly_charges": -100
    }
    p = predict(absurd_row)
    assert 0.0 <= p <= 1.0

def test_build_feature_dict_types():
    fd_dict = build_feature_dict(healthy_row)
    fd_obj = build_feature_dict(SimpleNamespace(**healthy_row))
    
    assert fd_dict == fd_obj
    
    for col in BASE_COLS:
        assert col in fd_dict

def test_band():
    from app.services.ml_model import _THRESH
    assert band(0.05) == "healthy"
    # test borderline cases per threshold
    assert band(_THRESH + 0.01) in ("at_risk", "critical")
    assert band(0.95) == "critical"

def test_explain():
    explanation = explain(critical_row, top_n=3)
    assert len(explanation) <= 3
    for ex in explanation:
        assert "feature" in ex
        assert "value" in ex
        assert "contribution" in ex
        assert "method" in ex
        assert ex["feature"] in _COLS

def test_predict_from_customer():
    res = predict_from_customer(SimpleNamespace(**healthy_row))
    assert "churn_probability" in res
    assert "band" in res
    assert "threshold_used" in res
    assert "top_features" in res
    assert "model_version" in res
    assert 0.0 <= res["churn_probability"] <= 1.0

def test_determinism():
    p1 = predict(mid_row)
    p2 = predict(mid_row.copy())
    assert p1 == p2  # Strict equality required

def test_db_consistency():
    db_path = os.path.join(os.path.dirname(__file__), "..", "churn_agent.db")
    if not os.path.exists(db_path):
        pytest.skip("churn_agent.db not found for consistency test")
        
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT * FROM customers LIMIT 200", conn)
    conn.close()
    
    if len(df) == 0:
        pytest.skip("No data in DB")
        
    df_churn = df[df["churn_label"] == 1].head(5)
    df_stay = df[df["churn_label"] == 0].head(5)
    
    if len(df_churn) == 0 or len(df_stay) == 0:
        pytest.skip("Not enough churn/stay examples in DB slice")
    
    probs_churn = [predict_from_customer(row)["churn_probability"] for _, row in df_churn.iterrows()]
    probs_stay = [predict_from_customer(row)["churn_probability"] for _, row in df_stay.iterrows()]
    
    mean_churn = sum(probs_churn) / len(probs_churn)
    mean_stay = sum(probs_stay) / len(probs_stay)
    
    assert mean_churn > mean_stay
