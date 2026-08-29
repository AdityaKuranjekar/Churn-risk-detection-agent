import pytest
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.services.action_catalog import pick_base_action, HIGH_ARR

def test_pick_base_action_healthy_monitor():
    signals = {
        "payment_failures": 0,
        "support_contacts": 0,
        "open_tickets": 0,
        "arr": 100,
        "usage_trend_pct": 0,
        "engagement_score": 100
    }
    action = pick_base_action(signals, 0.1)
    assert action["action"] == "monitor"
    assert action["priority"] == "P3"

def test_pick_base_action_billing_fix_priority():
    # Even if they have high churn prob, billing fix is evaluated first
    signals = {
        "payment_failures": 2,
        "arr": HIGH_ARR + 100
    }
    action = pick_base_action(signals, 0.9)
    assert action["action"] == "billing_fix_outreach"
    assert action["priority"] == "P0"

def test_pick_base_action_win_back():
    signals = {
        "payment_failures": 0,
        "arr": HIGH_ARR + 100
    }
    action = pick_base_action(signals, 0.85)
    assert action["action"] == "win_back_high_value"
    assert action["priority"] == "P0"

def test_pick_base_action_missing_keys():
    action = pick_base_action({}, 0.6)
    assert action["action"] == "csm_personal_checkin"
    assert action["priority"] == "P1"
