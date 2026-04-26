"""Smoke tests for the BYOK contract.

These tests verify that:
- /health is open (no key required)
- core endpoints return 401 when no X-Groq-Key header is present
- 401 responses carry a useful, actionable message (not a generic one)

The tests deliberately never supply a real Groq key — we're verifying
the authorization surface, not exercising the LLM pipeline.
"""

from app.main import app
from fastapi.testclient import TestClient

client = TestClient(app)


def test_health_open():
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "healthy"
    assert body["persistence"] == "stateless"


def test_optimize_without_key_returns_401():
    r = client.post(
        "/optimize_resume/",
        files={"resume_pdf": ("r.pdf", b"%PDF-1.4 fake", "application/pdf")},
        data={"jobdesc_text": "test"},
    )
    assert r.status_code == 401
    assert "X-Groq-Key" in r.json()["detail"]


def test_parse_without_key_returns_401():
    r = client.post(
        "/parse-resume/",
        files={"resume_pdf": ("r.pdf", b"%PDF-1.4 fake", "application/pdf")},
    )
    assert r.status_code == 401
    assert "X-Groq-Key" in r.json()["detail"]


def test_blank_key_header_rejected():
    r = client.post(
        "/optimize_resume/",
        files={"resume_pdf": ("r.pdf", b"%PDF-1.4 fake", "application/pdf")},
        data={"jobdesc_text": "test"},
        headers={"X-Groq-Key": "   "},
    )
    assert r.status_code == 401


def test_history_stateless_returns_501():
    r = client.get("/resume-history", headers={"X-User-ID": "u1"})
    assert r.status_code == 501
    assert "Supabase" in r.json()["detail"]
