"""HTTP API for the extension and the paste-text page.

* ``GET  /health``  readiness and index facts.
* ``POST /claims``  paragraphs -> check-worthy claims with offsets (fast, no models).
* ``POST /check``   claims (or raw text) -> Server-Sent Events, one ``verdict`` event per
  claim as soon as its batch is verified, then a ``done`` event with the page summary.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Callable, Sequence
from importlib import resources
from typing import Any, Protocol

import anyio
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field

from witness.checker import CheckResult
from witness.claims.extract import Claim, extract_claims
from witness.config import WitnessConfig
from witness.verify.verdict import summarize

_EVIDENCE_SHOWN = 3
_BATCH = 8


class CheckService(Protocol):
    def check_many(self, claims: Sequence[str]) -> list[CheckResult]: ...


class ClaimsRequest(BaseModel):
    title: str | None = Field(default=None, max_length=500)
    paragraphs: list[str] = Field(max_length=2000)


class ClaimIn(BaseModel):
    id: str = Field(max_length=64)
    text: str = Field(min_length=1, max_length=2000)
    query: str | None = Field(default=None, max_length=2000)


class CheckRequest(BaseModel):
    """Either ``claims`` (from ``/claims``) or ``text`` to extract claims from."""

    claims: list[ClaimIn] | None = Field(default=None, max_length=500)
    text: str | None = Field(default=None, max_length=100_000)
    title: str | None = Field(default=None, max_length=500)


def _claim_json(claim: Claim) -> dict[str, Any]:
    return {
        "id": claim.id, "paragraph": claim.paragraph, "start": claim.start,
        "end": claim.end, "text": claim.text, "query": claim.query,
    }  # fmt: skip


def verdict_json(claim_id: str, text: str, result: CheckResult) -> dict[str, Any]:
    d = result.decision
    return {
        "id": claim_id,
        "claim": text,
        "query": result.claim,
        "verdict": d.verdict,
        "confidence": round(d.confidence, 4),
        "label": d.fever_label,
        "probabilities": {
            "supports": round(d.probabilities[0], 4),
            "refutes": round(d.probabilities[1], 4),
            "not_enough_info": round(d.probabilities[2], 4),
        },
        "evidence": [
            {
                "title": e.title,
                "url": e.url,
                "sentence": e.text,
                "line": e.line,
                "score": round(e.score, 3),
            }
            for e in result.evidence[:_EVIDENCE_SHOWN]
        ],
        "timings_ms": {k: round(v, 1) for k, v in result.timings_ms.items()},
    }


def _sse(event: str, data: Any) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def create_app(
    config: WitnessConfig,
    checker_factory: Callable[[], CheckService],
    info: Callable[[], dict[str, Any]] | None = None,
) -> FastAPI:
    """Build the app; ``checker_factory`` runs once at startup (models load there)."""
    state: dict[str, CheckService] = {}

    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        state["checker"] = await anyio.to_thread.run_sync(checker_factory)
        yield

    app = FastAPI(title="witness", version="0.1.0", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origin_regex=r"^(chrome-extension://.*|http://(127\.0\.0\.1|localhost)(:\d+)?)$",
        allow_methods=["GET", "POST"],
        allow_headers=["content-type"],
    )

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok" if "checker" in state else "loading", **(info() if info else {})}

    @app.post("/claims")
    def claims(request: ClaimsRequest) -> dict[str, Any]:
        found, skipped = extract_claims(request.paragraphs, request.title, config.claims)
        return {
            "claims": [_claim_json(c) for c in found],
            "skipped": [
                {"paragraph": s.paragraph, "start": s.start, "end": s.end, "reason": s.reason}
                for s in skipped
            ],
        }

    @app.post("/check")
    async def check(request: CheckRequest) -> StreamingResponse:
        if request.claims is None and request.text is None:
            raise HTTPException(422, "send either `claims` or `text`")
        checker = state.get("checker")
        if checker is None:
            raise HTTPException(503, "models are still loading; retry in a few seconds")
        items: list[tuple[str, str, str]]
        preamble: list[str] = []
        if request.claims is not None:
            items = [(c.id, c.text, c.query or c.text) for c in request.claims]
        else:
            paragraphs = [p for p in (request.text or "").split("\n") if p.strip()]
            found, _ = extract_claims(paragraphs, request.title, config.claims)
            items = [(c.id, c.text, c.query) for c in found]
            preamble.append(_sse("claims", {"claims": [_claim_json(c) for c in found]}))

        async def stream() -> AsyncIterator[str]:
            for event in preamble:
                yield event
            verdicts: list[str] = []
            for start in range(0, len(items), _BATCH):
                batch = items[start : start + _BATCH]
                results = await anyio.to_thread.run_sync(
                    checker.check_many, [query for _, _, query in batch]
                )
                for (claim_id, text, _), result in zip(batch, results, strict=True):
                    verdicts.append(result.decision.verdict)
                    yield _sse("verdict", verdict_json(claim_id, text, result))
            summary = summarize(verdicts)  # type: ignore[arg-type]
            yield _sse("done", {"supported": summary.supported, "refuted": summary.refuted,
                                "unverifiable": summary.unverifiable})  # fmt: skip

        return StreamingResponse(
            stream(), media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )  # fmt: skip

    @app.get("/", response_class=HTMLResponse)
    def paste_page() -> str:
        return resources.files("witness.server").joinpath("static/index.html").read_text("utf-8")

    return app
