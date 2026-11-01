"""Drives the unpacked extension in headless Chromium against the API with fake models.

Run with ``uv run --group browser pytest -m browser`` after
``uv run --group browser playwright install chromium``.
"""

from __future__ import annotations

import functools
import http.server
import socket
import threading
import time
from collections.abc import Iterator
from pathlib import Path

import pytest
import uvicorn

from tests.fakes import FakeChecker
from witness.config import WitnessConfig
from witness.server.app import create_app

pytestmark = pytest.mark.browser
playwright_api = pytest.importorskip("playwright.sync_api")

ROOT = Path(__file__).resolve().parents[2]
EXTENSION = ROOT / "extension"
FIXTURES = Path(__file__).resolve().parent


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


@pytest.fixture(scope="module")
def api_url() -> Iterator[str]:
    port = _free_port()
    app = create_app(WitnessConfig(), FakeChecker)
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.time() + 10
    while not server.started and time.time() < deadline:
        time.sleep(0.05)
    yield f"http://127.0.0.1:{port}"
    server.should_exit = True
    thread.join(timeout=5)


@pytest.fixture(scope="module")
def page_url() -> Iterator[str]:
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(FIXTURES))
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}/fixture.html"
    httpd.shutdown()


def test_extension_underlines_claims_and_shows_evidence(
    api_url: str, page_url: str, tmp_path: Path
) -> None:
    with playwright_api.sync_playwright() as p:
        context = p.chromium.launch_persistent_context(
            str(tmp_path / "profile"),
            channel="chromium",
            headless=True,
            args=[
                f"--disable-extensions-except={EXTENSION}",
                f"--load-extension={EXTENSION}",
            ],
        )
        try:
            worker = context.service_workers[0] if context.service_workers else None
            worker = worker or context.wait_for_event("serviceworker", timeout=10_000)
            extension_id = worker.url.split("/")[2]

            options = context.new_page()
            options.goto(f"chrome-extension://{extension_id}/options.html")
            options.fill("#server", api_url)
            options.click("#save")
            options.wait_for_selector("#result:has-text('Saved')")
            options.close()

            page = context.new_page()
            page.goto(page_url)
            page.wait_for_selector("span.witness-refuted", timeout=15_000)
            page.wait_for_function(
                "document.querySelectorAll('span.witness-pending').length === 0", timeout=15_000
            )

            def verdict_text(kind: str) -> list[str]:
                texts = page.eval_on_selector_all(
                    f"span.witness-{kind}", "els => els.map(e => e.textContent)"
                )
                return [str(t) for t in texts]

            supported = "".join(verdict_text("supported"))
            assert "Paris is the capital of France." in supported
            # A claim spanning a link is wrapped piecewise and keeps the link intact.
            assert "Berlin is the capital of Germany" in supported
            assert page.locator("#p2 a").inner_text() == "capital"
            assert verdict_text("refuted") == ["Paris was never part of the Roman Empire."]
            assert verdict_text("unverifiable") == ["Paris hosted the Summer Olympics in 1900."]
            # Questions, navigation and boilerplate are never underlined.
            underlined = page.eval_on_selector_all(
                "nav span.witness-claim, footer span.witness-claim", "els => els.length"
            )
            assert underlined == 0
            assert "What do you think" not in "".join(
                page.eval_on_selector_all(
                    "span.witness-claim", "els => els.map(e => e.textContent)"
                )
            )
            # The page text itself is unchanged by the underlines.
            assert page.inner_text("#p1").startswith("Paris is the capital of France. Paris was")

            page.hover("span.witness-refuted")
            card = page.locator("#witness-card-host .card")
            card.wait_for(state="visible", timeout=5_000)
            assert "refuted" in card.inner_text().lower()
            assert "% confidence" in card.inner_text()
            link = card.locator("a").first
            assert link.get_attribute("href") == "https://en.wikipedia.org/wiki/Paris"
        finally:
            context.close()
