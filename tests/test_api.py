"""API tests that run on any laptop/CPU: start the mock first (uvicorn tests.mock_openai:app --port 8000)."""
import os
import sys

sys.path[:0] = ["api", "scripts"]                 # import app.py and common.py
os.environ.setdefault("VLLM_URL", "http://localhost:8000/v1")
os.environ["API_KEY"] = "test-key"

from fastapi.testclient import TestClient          # noqa: E402
import app as api                                  # noqa: E402

client = TestClient(api.app)
HEADERS = {"X-API-Key": "test-key"}


def test_health():
    assert client.get("/health").json() == {"status": "ok"}


def test_ready_lists_adapters():
    body = client.get("/ready").json()
    assert body["status"] == "ready" and "qa-dpo" in body["models"]


def test_qa_routes_to_dpo_adapter():
    r = client.post("/v1/generate", headers=HEADERS, json={"task": "qa", "input": "Q?\n\nOptions:\nA) x\nB) y"})
    assert r.status_code == 200
    assert r.json()["model"] == "qa-dpo" and r.json()["output"].startswith("Answer:")


def test_summary_returns_parsed_json():
    r = client.post("/v1/generate", headers=HEADERS, json={"task": "summary", "input": "A 40-year-old ..."})
    assert r.json()["parsed_json"]["patient information"]["age"] == "40"


def test_auth_and_validation():
    assert client.post("/v1/generate", json={"task": "qa", "input": "x"}).status_code == 401
    assert client.post("/v1/generate", headers=HEADERS, json={"task": "bad", "input": "x"}).status_code == 422
