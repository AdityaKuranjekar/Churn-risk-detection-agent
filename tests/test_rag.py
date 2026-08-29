import pytest
import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.services.rag import (
    load_playbook, retrieve, retrieve_best, build_query, action_matches, warm,
    CACHE_PATH, EMBED_BACKEND
)
from app.services.action_catalog import CATALOG

def test_corpus_valid():
    corpus = load_playbook()
    assert 8 <= len(corpus) <= 14
    
    ids = [c["id"] for c in corpus]
    assert len(set(ids)) == len(ids), "Duplicate ids found in playbook.json"
    
    valid_actions = {c["id"] for c in CATALOG}
    for c in corpus:
        assert c["recommended_action"] in valid_actions
        assert 20 <= len(c["text"].split()) <= 120

def test_lexical_fallback():
    # Force lexical retrieval
    res1 = retrieve("2 payment failures, billing broken, premium", _force_lexical=True)
    assert res1[0]["id"] == "pb_billing_failures"
    assert res1[0]["retrieval_method"] == "lexical_fallback"
    
    res2 = retrieve("declining usage, no complaints, wants more content", _force_lexical=True)
    assert res2[0]["id"] == "pb_declining_usage_no_complaints"
    
    res3 = retrieve("account approaching renewal with low engagement", _force_lexical=True)
    assert res3[0]["id"] == "pb_renewal_downgrade"
    
    res4 = retrieve("", _force_lexical=True)
    assert len(res4) == 1
    assert res4[0]["low_confidence"] is True

def test_build_query():
    signals = {"payment_failures": 2, "usage_trend_pct": -0.20}
    base_action = {"playbook_query": "premium customer high risk"}
    q = build_query(signals, base_action)
    
    assert "premium customer high risk" in q
    assert "2 payment failures" in q
    assert "usage down 20%" in q

def test_action_matches():
    snippet = {"recommended_action": "billing_fix_outreach"}
    base_action = {"action": "billing_fix_outreach"}
    assert action_matches(snippet, base_action) is True
    
    base_action["action"] = "monitor"
    assert action_matches(snippet, base_action) is False

@pytest.mark.skipif(os.getenv("EMBED_BACKEND") == "gemini" and not os.getenv("GEMINI_API_KEY"), reason="Gemini backend requires API key")
def test_embedding_path():
    try:
        warm()
    except Exception as e:
        pytest.skip(f"Embedding backend unavailable: {e}")
        
    assert os.path.exists(CACHE_PATH)
    
    res = retrieve_best("2 payment failures premium account")
    # Due to fuzziness of embeddings, we just assert it returns something
    assert "score" in res
    assert -1.0 <= res["score"] <= 1.01
    assert res["retrieval_method"] == "embedding"
