import pytest
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.services.scoring import risk_breakdown, agreement, BASELINE

def test_risk_breakdown_healthy():
    signals = {
        "usage_trend_pct": 0.20,
        "last_login_days": 1,
        "engagement_score": 85,
        "support_contacts": 0,
        "open_tickets": 0,
        "avg_sentiment": 0.6,
        "payment_failures": 0,
        "tenure_days": 1200,
        "days_to_renewal": 200
    }
    res = risk_breakdown(signals)
    assert res["health_score"] > 90
    assert res["risk_band"] == "healthy"
    assert len(res["contributors"]) == 0
    assert len(res["positives"]) > 0

def test_risk_breakdown_critical():
    signals = {
        "usage_trend_pct": -0.50,
        "last_login_days": 35,
        "engagement_score": 15,
        "support_contacts": 4,
        "open_tickets": 1,
        "avg_sentiment": -0.8,
        "payment_failures": 2,
        "tenure_days": 150,
        "days_to_renewal": 20
    }
    res = risk_breakdown(signals)
    assert res["health_score"] < 40
    assert res["risk_band"] == "critical"
    assert len(res["contributors"]) > 0

def test_risk_breakdown_missing_keys():
    res = risk_breakdown({})
    # With missing keys, it should fall back to neutral/zeros, no crashes
    assert res["health_score"] <= BASELINE
    assert res["health_score"] >= 0

def test_agreement():
    # Health 80 (implied risk 0.2), ML prob 0.1 => gap 0.1 => agree
    res = agreement(80, 0.1)
    assert res["agree"] is True
    
    # Health 80 (implied risk 0.2), ML prob 0.9 => gap 0.7 => disagree
    res = agreement(80, 0.9)
    assert res["agree"] is False
    
    # Health 20 (implied risk 0.8), ML prob 0.9 => gap 0.1 => agree
    res = agreement(20, 0.9)
    assert res["agree"] is True
