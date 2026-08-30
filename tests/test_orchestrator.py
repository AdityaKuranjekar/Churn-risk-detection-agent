import os
import pytest
from app.db import Base, engine, get_session, SessionLocal
from app.models import Customer, UsageDaily, Feedback, Analysis, Meta
from app.services import signals, orchestrator

# Ensure we're using a test DB or something safe, 
# but per instructions, we can just use the real DB (read-only for signals) 
# and rollback transactions to avoid polluting history.

@pytest.fixture(scope="module")
def setup_test_db():
    Base.metadata.create_all(engine)
    yield
    # Could drop tables here if it were a purely temp DB, but we reuse the dev DB

@pytest.fixture
def session():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.rollback()
        db.close()

def test_trend():
    # synthetic declining
    class MockRow:
        def __init__(self, am):
            self.active_minutes = am
    
    # 30 base rows of 100
    # 7 recent rows of 50
    rows = [MockRow(100) for _ in range(30)] + [MockRow(50) for _ in range(7)]
    assert signals._trend(rows) == -0.5

    # synthetic flat
    rows = [MockRow(100) for _ in range(37)]
    assert signals._trend(rows) == 0.0

    # < 14 rows
    rows = [MockRow(100) for _ in range(13)]
    assert signals._trend(rows) == 0.0

def test_get_signals(session):
    # Just need an existing customer
    # We don't know the exact ID, so let's pick one that exists
    c = session.query(Customer).first()
    if not c:
        pytest.skip("No customers in DB")
    
    sig = signals.get_signals(session, c.id)
    assert "customer_id" in sig
    assert "usage_series" in sig
    
    with pytest.raises(signals.CustomerNotFound):
        signals.get_signals(session, 999999)

def test_analyze_customer_healthy(monkeypatch, session):
    # Find a healthy customer or mock the dependencies
    # We will mock the ML and scoring directly to force the low-risk path
    
    c = session.query(Customer).first()
    if not c:
        pytest.skip("No customers in DB")
        
    def mock_predict(*args):
        return {"churn_probability": 0.10, "band": "low", "top_features": [], "model_version": "v1"}
        
    def mock_breakdown(*args):
        return {"health_score": 85, "risk_band": "healthy", "contributors": [], "positives": []}
        
    monkeypatch.setattr(orchestrator.ml_model, "predict_from_customer", mock_predict)
    monkeypatch.setattr(orchestrator.scoring, "risk_breakdown", mock_breakdown)
    
    # Ensure LLM is NOT called
    def mock_llm(*args):
        raise Exception("LLM should not be called on low risk!")
    monkeypatch.setattr(orchestrator.llm, "analyze", mock_llm)
    
    res = orchestrator.analyze_customer(session, c.id, force=True)
    assert res["generated_by"] == "low_risk_shortcut"
    assert res["playbook"]["id"] == ""
    assert res["draft_message"] == ""

def test_analyze_customer_critical(monkeypatch, session):
    c = session.query(Customer).first()
    if not c:
        pytest.skip("No customers in DB")
        
    def mock_predict(*args):
        return {"churn_probability": 0.90, "band": "critical", "top_features": [], "model_version": "v1"}
        
    def mock_breakdown(*args):
        return {"health_score": 20, "risk_band": "critical", "contributors": [], "positives": []}
        
    monkeypatch.setattr(orchestrator.ml_model, "predict_from_customer", mock_predict)
    monkeypatch.setattr(orchestrator.scoring, "risk_breakdown", mock_breakdown)
    
    # disable real LLM call to save time/cost, use LLM_DISABLE=1 fallback logic
    monkeypatch.setattr(orchestrator.llm, "LLM_ENABLED", False)
    
    res = orchestrator.analyze_customer(session, c.id, force=True)
    assert res["generated_by"] == "fallback"
    assert "analysis_id" in res
    assert res["escalate"] == (res["priority"] == "P0")

def test_rag_monkeypatch_fail(monkeypatch, session):
    c = session.query(Customer).first()
    if not c:
        pytest.skip("No customers")

    def mock_predict(*args):
        return {"churn_probability": 0.90, "band": "critical", "top_features": [], "model_version": "v1"}
    monkeypatch.setattr(orchestrator.ml_model, "predict_from_customer", mock_predict)
    
    def mock_rag(*args):
        raise Exception("RAG failure")
    monkeypatch.setattr(orchestrator.rag, "build_query", mock_rag)
    
    monkeypatch.setattr(orchestrator.llm, "LLM_ENABLED", False)
    res = orchestrator.analyze_customer(session, c.id, force=True)
    # Should not throw exception, should return snippet empty
    assert res["playbook"]["id"] == ""

def test_list_dashboard_rows(session):
    rows = orchestrator.list_dashboard_rows(session, demo_only=True)
    assert isinstance(rows, list)
    if len(rows) > 1:
        assert rows[0]["churn_probability"] >= rows[-1]["churn_probability"]

def test_approve_analysis(session):
    from app.models import Analysis
    a = Analysis(
        customer_id=1,
        churn_probability=0.5,
        health_score=50,
        risk_band="watch",
        priority="P2",
        recommended_action="None",
        generated_by="test",
        result_json="{}"
    )
    session.add(a)
    session.commit()
    session.refresh(a)
    res = orchestrator.approve_analysis(session, a.id, "edited msg", "approved")
    assert res["status"] == "approved"
    assert res["approved_message"] == "edited msg"