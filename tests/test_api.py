import pytest
from fastapi.testclient import TestClient
from app.main import app

@pytest.fixture(scope="module")
def client():
    # triggers lifespan (rag.warm, init_db)
    with TestClient(app) as c:
        yield c

def test_health(client, monkeypatch):
    from app.services import llm
    monkeypatch.setattr(llm, "LLM_ENABLED", False)
    resp = client.get("/api/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["components"]["ml"]["loaded"] is True
    assert data["components"]["llm"]["enabled"] is False

def test_customers_list(client):
    resp = client.get("/api/customers?demo_only=true")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)
    assert len(data) <= 40
    if len(data) > 1:
        assert data[0]["churn_probability"] >= data[-1]["churn_probability"]

def test_customers_list_all(client):
    resp = client.get("/api/customers?demo_only=false")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)

def test_customer_detail_missing(client):
    resp = client.get("/api/customers/999999")
    assert resp.status_code == 404

def test_customer_detail(client):
    # Find a valid customer
    resp = client.get("/api/customers?demo_only=true")
    c_id = resp.json()[0]["customer_id"]
    
    resp2 = client.get(f"/api/customers/{c_id}")
    assert resp2.status_code == 200
    data = resp2.json()
    assert "customer" in data
    assert "signals" in data
    assert len(data["usage_series"]) > 0

def test_analyze_critical(client, monkeypatch):
    from app.services import llm
    monkeypatch.setattr(llm, "LLM_ENABLED", False)
    
    # Get critical customer (top of list)
    resp = client.get("/api/customers?demo_only=true")
    c_id = resp.json()[0]["customer_id"]
    
    resp2 = client.post(f"/api/customers/{c_id}/analyze")
    assert resp2.status_code == 200
    data = resp2.json()
    assert data["priority"] in ["P0", "P1", "P2", "P3"]
    assert data["generated_by"] == "fallback"
    assert "analysis_id" in data
    assert data["escalate"] == (data["priority"] == "P0")
    
    # Test second analyze to ensure new history
    a_id1 = data["analysis_id"]
    resp3 = client.post(f"/api/customers/{c_id}/analyze")
    a_id2 = resp3.json()["analysis_id"]
    assert a_id1 != a_id2
    
    # GET detail should show the last analysis
    resp4 = client.get(f"/api/customers/{c_id}")
    assert resp4.json()["last_analysis"]["analysis_id"] == a_id2

def test_analyze_healthy(client, monkeypatch):
    from app.services import llm
    monkeypatch.setattr(llm, "LLM_ENABLED", False)
    # Get a healthy customer (bottom of list)
    resp = client.get("/api/customers?demo_only=true")
    c_id = resp.json()[-1]["customer_id"]
    
    resp2 = client.post(f"/api/customers/{c_id}/analyze")
    assert resp2.status_code == 200
    data = resp2.json()
    assert "generated_by" in data

def test_approve_analysis(client, monkeypatch):
    from app.services import llm
    monkeypatch.setattr(llm, "LLM_ENABLED", False)
    
    resp = client.get("/api/customers?demo_only=true")
    c_id = resp.json()[0]["customer_id"]
    
    resp2 = client.post(f"/api/customers/{c_id}/analyze")
    a_id = resp2.json()["analysis_id"]
    
    resp3 = client.post(f"/api/customers/{c_id}/approve", json={"status": "approved", "message": "ok", "analysis_id": a_id})
    assert resp3.status_code == 200
    data = resp3.json()
    assert data["status"] == "approved"
    
    resp4 = client.get(f"/api/customers/{c_id}")
    assert resp4.json()["last_analysis"]["status"] == "approved"

def test_approve_validation_error(client):
    resp = client.get("/api/customers?demo_only=true")
    c_id = resp.json()[0]["customer_id"]
    
    resp2 = client.post(f"/api/customers/{c_id}/approve", json={"status": "garbage", "message": "ok"})
    assert resp2.status_code == 422

def test_analyze_missing(client):
    resp = client.post("/api/customers/999999/analyze")
    assert resp.status_code == 404

def test_openapi(client):
    resp = client.get("/openapi.json")
    assert resp.status_code == 200

@pytest.mark.xfail(reason="Auth disabled in step 8, wait for step 11")
def test_auth_failure(monkeypatch):
    monkeypatch.setenv("AUTH_DISABLED", "0")
    # Need a fresh client to pick up the env var
    with TestClient(app) as c:
        resp = c.get("/api/health")
        assert resp.status_code == 401
