import pytest
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.services.llm import safe_json_parse, _validate_and_coerce, analyze
from app.services.prompts import build_user_prompt

# Fixtures
@pytest.fixture
def base_inputs():
    return {
        "signals": {
            "name": "Alice",
            "plan_tier": "Premium",
            "tenure_days": 150,
            "usage_trend_pct": -0.2
        },
        "churn_prob": 0.85,
        "breakdown": {
            "health_score": 30,
            "risk_band": "at_risk",
            "contributors": [
                {"label": "Declining Usage", "detail": "Usage dropped 20%"}
            ],
            "positives": []
        },
        "playbook_snippet": {
            "title": "Re-engagement Playbook",
            "text": "Call the customer.",
            "retrieval_method": "embedding",
            "low_confidence": False
        },
        "base_action": {
            "action_label": "Executive Check-in",
            "priority": "P1",
            "channel": "call",
            "rationale": "High drop in usage."
        }
    }

# PARSE / COERCE
def test_safe_json_parse():
    assert safe_json_parse('```json\n{"a":1}\n```') == {"a": 1}
    assert safe_json_parse('garbage {"summary":"x"} trailing') == {"summary": "x"}
    assert safe_json_parse('not json at all') is None

def test_validate_and_coerce(base_inputs):
    # Wrong priority
    obj = {
        "summary": "x",
        "top_reasons": ["a", "b", "c", "d", "e"],
        "recommended_action": "Completely unrelated action",
        "priority": "P3",
        "draft_message": "",
        "playbook_citation": ""
    }
    
    out = _validate_and_coerce(obj, base_inputs["base_action"], base_inputs["playbook_snippet"])
    
    assert out["priority"] == "P1"  # overriden
    assert "Executive Check-in" in out["recommended_action"]  # re-anchored
    assert len(out["top_reasons"]) == 3  # truncated
    assert out["draft_message"] != ""  # templated fallback
    assert out["playbook_citation"] == "Re-engagement Playbook"

# PROMPT
def test_build_user_prompt(base_inputs):
    prompt = build_user_prompt(**base_inputs)
    assert "Re-engagement Playbook" in prompt
    assert "Executive Check-in" in prompt
    assert "85%" in prompt
    assert "Alice" in prompt
    assert "Premium" in prompt
    assert "150" in prompt
    
    # Missing fields
    base_inputs["signals"] = {"name": "Bob"}
    prompt2 = build_user_prompt(**base_inputs)
    assert "Bob" in prompt2

# OFFLINE (fallback)
def test_analyze_offline(base_inputs, monkeypatch):
    monkeypatch.setenv("LLM_DISABLE", "1")
    # Force reload of LLM_ENABLED if necessary, but actually analyze() checks bool(LLM_ENABLED).
    # We can mock it inside the module for the test.
    import app.services.llm as llm
    monkeypatch.setattr(llm, "LLM_ENABLED", False)
    
    out = analyze(**base_inputs)
    assert out["priority"] == "P1"
    assert out["recommended_action"] == "Executive Check-in"
    assert isinstance(out["top_reasons"], list)
    assert len(out["top_reasons"]) == 1
    assert out["playbook_citation"] == "Re-engagement Playbook"
    assert out["_generated_by"] == "fallback"
    assert "Alice" in out["draft_message"]
    assert "[" not in out["draft_message"]
    
    # Low confidence edge case
    base_inputs["playbook_snippet"]["low_confidence"] = True
    out_low = analyze(**base_inputs)
    assert out_low["_generated_by"] == "fallback"
    
    # Zero contributors edge case
    base_inputs["breakdown"]["contributors"] = []
    out_zero = analyze(**base_inputs)
    assert out_zero["_generated_by"] == "fallback"
    assert "overall usage" in out_zero["summary"]

# LIVE (opt-in)
@pytest.mark.skipif(os.getenv("RUN_LIVE_LLM") != "1", reason="Live LLM tests require RUN_LIVE_LLM=1")
def test_analyze_live(base_inputs):
    import app.services.llm as llm
    if not llm.GEMINI_API_KEY or llm.GEMINI_API_KEY.startswith("PAIza"):
        pytest.skip("No real Gemini API key")
        
    out = analyze(**base_inputs)
    assert out["priority"] == "P1"
    assert out["_generated_by"].startswith("gemini")
    assert "summary" in out
    # Usually the prompt tells it to cite the playbook title
    assert "Re-engagement" in out["summary"] or "Re-engagement" in out["playbook_citation"]
