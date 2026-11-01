"""Record the README hero GIF: the unpacked extension checking the demo article.

Needs a running ``witness serve`` (real index and models) on the configured port, and
Chromium for Playwright. Frames are screenshots of the real page while verdicts stream
in, then while hovering one refuted and one supported claim.

    uv run --group docs python scripts/record_demo.py --server http://127.0.0.1:8094
"""

from __future__ import annotations

import argparse
import functools
import http.server
import io
import tempfile
import threading
import time
from pathlib import Path

from PIL import Image
from playwright.sync_api import Page, ViewportSize, sync_playwright

ROOT = Path(__file__).resolve().parents[1]
EXTENSION = ROOT / "extension"
DEMO_DIR = ROOT / "docs" / "demo"
VIEWPORT: ViewportSize = {"width": 1100, "height": 760}


def _serve_demo() -> tuple[http.server.ThreadingHTTPServer, str]:
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(DEMO_DIR))
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, f"http://127.0.0.1:{httpd.server_address[1]}/marie-curie.html"


def _frame(page: Page) -> Image.Image:
    return Image.open(io.BytesIO(page.screenshot())).convert("RGB")


def record(server: str, out_gif: Path, out_png: Path) -> None:
    httpd, url = _serve_demo()
    frames: list[tuple[Image.Image, int]] = []
    with sync_playwright() as p, tempfile.TemporaryDirectory() as profile:
        context = p.chromium.launch_persistent_context(
            profile,
            channel="chromium",
            headless=True,
            viewport=VIEWPORT,
            device_scale_factor=1,
            args=[f"--disable-extensions-except={EXTENSION}", f"--load-extension={EXTENSION}"],
        )
        worker = context.service_workers[0] if context.service_workers else None
        worker = worker or context.wait_for_event("serviceworker", timeout=10_000)
        extension_id = worker.url.split("/")[2]
        options = context.new_page()
        options.goto(f"chrome-extension://{extension_id}/options.html")
        options.fill("#server", server)
        options.click("#save")
        options.close()

        page = context.new_page()
        page.set_viewport_size(VIEWPORT)
        page.goto(url)
        frames.append((_frame(page), 900))
        started = time.time()
        while time.time() - started < 60:
            frames.append((_frame(page), 350))
            pending = page.eval_on_selector_all("span.witness-pending", "els => els.length")
            total = page.eval_on_selector_all("span.witness-claim", "els => els.length")
            if total and not pending:
                break
            time.sleep(0.3)
        frames.append((_frame(page), 1500))
        hovered: list[str] = []
        for verdict in ("refuted", "supported", "unverifiable"):
            spans = page.locator(f"span.witness-{verdict}")
            if spans.count() == 0:
                continue
            spans.first.hover()
            page.wait_for_timeout(400)
            frames.append((_frame(page), 3200))
            hovered.append(verdict)
            page.mouse.move(5, 5)
            page.wait_for_timeout(400)
        if not hovered:
            raise RuntimeError("no verdicts arrived; is the server running with real models?")
        last = page.locator(f"span.witness-{hovered[0]}")
        last.nth(min(1, last.count() - 1)).hover()
        page.wait_for_timeout(400)
        still = _frame(page)
        frames.append((still, 3200))
        still.save(out_png)
        context.close()
    httpd.shutdown()

    images = [f.quantize(colors=128, method=Image.Quantize.MEDIANCUT) for f, _ in frames]
    images[0].save(
        out_gif,
        save_all=True,
        append_images=images[1:],
        duration=[d for _, d in frames],
        loop=0,
        optimize=True,
    )
    size_mb = out_gif.stat().st_size / 1e6
    print(f"wrote {out_gif} ({len(frames)} frames, {size_mb:.1f} MB) and {out_png}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", default="http://127.0.0.1:8094")
    parser.add_argument("--gif", type=Path, default=ROOT / "docs/assets/demo.gif")
    parser.add_argument("--png", type=Path, default=ROOT / "docs/assets/hover-card.png")
    args = parser.parse_args()
    args.gif.parent.mkdir(parents=True, exist_ok=True)
    record(args.server, args.gif, args.png)


if __name__ == "__main__":
    main()
