from __future__ import annotations

import json
from typing import Any

from fastapi.testclient import TestClient

from tests.fakes import FakeChecker
from witness.config import WitnessConfig
from witness.server.app import create_app


def _client() -> TestClient:
    return TestClient(create_app(WitnessConfig(), FakeChecker, lambda: {"index": "fake"}))


def _events(body: str) -> list[tuple[str, Any]]:
    out = []
    for block in body.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.split("\n"))
        out.append((lines["event"], json.loads(lines["data"])))
    return out


def test_health_reports_ready_after_startup() -> None:
    with _client() as client:
        body = client.get("/health").json()
    assert body == {"status": "ok", "index": "fake"}


def test_claims_endpoint_returns_offsets() -> None:
    paragraphs = ["Paris is the capital of France. Do you like it? It has many famous bridges."]
    with _client() as client:
        body = client.post("/claims", json={"title": "Paris", "paragraphs": paragraphs}).json()
    claims = body["claims"]
    assert [c["text"] for c in claims] == ["Paris is the capital of France.",
                                           "It has many famous bridges."]  # fmt: skip
    assert claims[1]["query"] == "Paris has many famous bridges."
    for c in claims:
        assert paragraphs[0][c["start"] : c["end"]] == c["text"]
    assert body["skipped"][0]["reason"] == "question"


def test_check_streams_one_verdict_per_claim_then_done() -> None:
    claims = [
        {"id": "a", "text": "Paris is the capital of France."},
        {"id": "b", "text": "Paris is not in Europe."},
        {"id": "c", "text": "Paris hosted the 1900 Olympics."},
    ]
    with _client() as client:
        response = client.post("/check", json={"claims": claims})
    assert response.headers["content-type"].startswith("text/event-stream")
    events = _events(response.text)
    verdicts = {d["id"]: d["verdict"] for e, d in events if e == "verdict"}
    assert verdicts == {"a": "supported", "b": "refuted", "c": "unverifiable"}
    assert events[-1] == ("done", {"supported": 1, "refuted": 1, "unverifiable": 1})
    first = next(d for e, d in events if e == "verdict")
    assert first["evidence"][0]["url"].startswith("https://en.wikipedia.org/wiki/")
    assert 0.0 <= first["confidence"] <= 1.0


def test_check_from_raw_text_announces_claims_first() -> None:
    with _client() as client:
        response = client.post("/check", json={"text": "Paris is the capital of France."})
    events = _events(response.text)
    assert events[0][0] == "claims"
    assert events[0][1]["claims"][0]["text"] == "Paris is the capital of France."
    assert events[1][0] == "verdict"


def test_check_rejects_empty_requests() -> None:
    with _client() as client:
        assert client.post("/check", json={}).status_code == 422


def test_paste_page_is_served() -> None:
    with _client() as client:
        page = client.get("/")
    assert page.status_code == 200
    assert "paste" in page.text.lower()


def test_cors_allows_the_extension_origin_only() -> None:
    with _client() as client:
        ok = client.options(
            "/claims",
            headers={
                "Origin": "chrome-extension://abcdef",
                "Access-Control-Request-Method": "POST",
            },
        )
        bad = client.options(
            "/claims",
            headers={"Origin": "https://evil.example", "Access-Control-Request-Method": "POST"},
        )
    assert ok.headers.get("access-control-allow-origin") == "chrome-extension://abcdef"
    assert "access-control-allow-origin" not in bad.headers
