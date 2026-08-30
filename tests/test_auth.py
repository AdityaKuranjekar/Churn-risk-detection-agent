import os
import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

@pytest.fixture(autouse=True)
def setup_env():
    os.environ["APP_USERNAME"] = "testuser"
    os.environ["APP_PASSWORD"] = "testpass"
    os.environ["AUTH_DISABLED"] = "0"
    yield
    os.environ.pop("APP_USERNAME", None)
    os.environ.pop("APP_PASSWORD", None)
    os.environ.pop("AUTH_DISABLED", None)

def test_auth_flow():
    # 1. /api/me should 401 initially when AUTH_DISABLED=0
    r = client.get("/api/me")
    assert r.status_code == 401

    # 2. Login with bad creds -> 401
    r = client.post("/api/login", json={"username": "testuser", "password": "bad"})
    assert r.status_code == 401

    # 3. Login with good creds -> 200
    r = client.post("/api/login", json={"username": "testuser", "password": "testpass"})
    assert r.status_code == 200
    assert r.json() == {"user": "testuser"}

    # 4. /api/me should return user
    r = client.get("/api/me")
    assert r.status_code == 200
    assert r.json() == {"user": "testuser"}

    # 5. Logout -> /api/me 401
    r = client.post("/api/logout")
    assert r.status_code == 200
    r = client.get("/api/me")
    assert r.status_code == 401
