"""Command-line interface: ``witness build-index | check | check-url | eval | serve``."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer

from witness.config import load_config
from witness.errors import WitnessError

if TYPE_CHECKING:
    from witness.checker import CheckResult

app = typer.Typer(add_completion=False, no_args_is_help=True, help=__doc__)

ConfigOpt = Annotated[Path | None, typer.Option("--config", "-c", help="TOML config file.")]


@app.callback()
def _root() -> None:
    """witness: underline every claim on a page, green / red / grey, with Wikipedia evidence."""


def _setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    for noisy in ("httpx", "httpcore", "huggingface_hub", "filelock"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


@app.command("build-index")
def build_index_cmd(
    dump: Annotated[Path, typer.Option(help="FEVER wiki-pages.zip")] = Path(
        "data/raw/wiki-pages.zip"
    ),
    out: Annotated[Path | None, typer.Option(help="Output directory (default: config)")] = None,
    lite: Annotated[
        int, typer.Option(help="Keep only the N most-linked pages (0 = full dump)")
    ] = 0,
    workers: Annotated[int, typer.Option(help="Parser processes")] = 7,
    limit_shards: Annotated[int | None, typer.Option(help="Smoke test on the first shards")] = None,
    config: ConfigOpt = None,
) -> None:
    """Build the page store, alias table and BM25 index from the FEVER dump."""
    from witness.wiki.build import build_index, most_linked_titles

    _setup_logging()
    cfg = load_config(config)
    keep = most_linked_titles(dump, lite, workers) if lite else None
    stats = build_index(
        dump, out or cfg.index_dir, workers=workers, keep=keep, limit_shards=limit_shards
    )
    typer.echo(json.dumps(stats, indent=2))


_COLORS = {"supported": typer.colors.GREEN, "refuted": typer.colors.RED,
           "unverifiable": typer.colors.BRIGHT_BLACK}  # fmt: skip


def _print_result(result: CheckResult) -> None:
    d = result.decision
    line = f"{d.verdict.upper():<13} {d.confidence:5.0%}  {result.claim}"
    typer.secho(line, fg=_COLORS[d.verdict])
    for e in result.evidence[:3]:
        typer.echo(f"    [{e.title}:{e.line}] {e.text}")


@app.command("check")
def check_cmd(
    claims: Annotated[list[str], typer.Argument(help="One or more claims to verify")],
    as_json: Annotated[bool, typer.Option("--json", help="Machine-readable output")] = False,
    config: ConfigOpt = None,
) -> None:
    """Verify claims against the Wikipedia snapshot."""
    from witness.checker import build_checker
    from witness.server.app import verdict_json

    checker = build_checker(load_config(config))
    for i, result in enumerate(checker.check_many(claims)):
        if as_json:
            typer.echo(json.dumps(verdict_json(f"c{i}", result.claim, result)))
        else:
            _print_result(result)


@app.command("check-url")
def check_url_cmd(
    url: Annotated[str, typer.Argument(help="http(s) URL of an article")],
    config: ConfigOpt = None,
) -> None:
    """Fetch a page, extract its check-worthy claims and verify them."""
    import httpx

    from witness.checker import build_checker
    from witness.claims.extract import extract_claims
    from witness.claims.readable import readable_text

    if not url.startswith(("http://", "https://")):
        raise typer.BadParameter("url must start with http:// or https://")
    cfg = load_config(config)
    try:
        response = httpx.get(url, follow_redirects=True, timeout=20.0,
                             headers={"User-Agent": "witness-fact-checker/0.1"})  # fmt: skip
        response.raise_for_status()
    except httpx.HTTPError as exc:
        raise WitnessError(f"could not fetch {url}: {exc}") from exc
    page = readable_text(response.text)
    found, skipped = extract_claims(page.paragraphs, page.title, cfg.claims)
    typer.echo(f"{page.title or url}: {len(found)} claims, {len(skipped)} sentences skipped")
    checker = build_checker(cfg)
    results = checker.check_many([c.query for c in found])
    for result in results:
        _print_result(result)
    counts = {v: sum(r.decision.verdict == v for r in results) for v in _COLORS}
    typer.echo(json.dumps(counts))


@app.command("serve")
def serve_cmd(config: ConfigOpt = None) -> None:
    """Run the HTTP API (used by the browser extension) and the paste-text page."""
    import uvicorn

    from witness.checker import build_checker
    from witness.server.app import create_app

    _setup_logging()
    cfg = load_config(config)

    def info() -> dict[str, object]:
        return {
            "index": str(cfg.index_dir),
            "verifier": cfg.verifier.model,
            "reranker": cfg.reranker.model,
            "snapshot": "Wikipedia, June 2017 (FEVER)",
        }

    # Load the index and models before uvicorn starts: a missing index or verifier then
    # ends in one actionable `error:` line (see main) instead of a startup traceback.
    checker = build_checker(cfg)
    app_ = create_app(cfg, lambda: checker, info)
    uvicorn.run(app_, host=cfg.server.host, port=cfg.server.port, log_level="info")


@app.command("train-verifier")
def train_verifier_cmd(
    stage: Annotated[
        str, typer.Option(help="examples (CPU: build training data) or train (GPU)")
    ] = "examples",
    per_label: Annotated[int, typer.Option(help="Training claims per label")] = 17000,
    epochs: Annotated[int, typer.Option()] = 1,
    examples_file: Annotated[Path, typer.Option(help="Training examples (JSONL)")] = Path(
        "runs/verifier_examples.jsonl"
    ),
    out: Annotated[Path, typer.Option(help="Where to save the model")] = Path("artifacts/verifier"),
    raw_dir: Annotated[Path, typer.Option(help="Folder with train.jsonl")] = Path("data/raw"),
    workers: Annotated[int, typer.Option()] = 6,
    config: ConfigOpt = None,
) -> None:
    """Fine-tune the DeBERTa-v3 verifier on a balanced FEVER train sample.

    Two stages so the GPU is only held while it is actually used.
    """
    import dataclasses
    import time

    from witness.verify.train_data import Example

    _setup_logging()
    cfg = load_config(config)
    if stage == "examples":
        from witness.evaluation.commands import make_examples
        from witness.evaluation.fever_data import load_claims, split_train
        from witness.verify.train_data import balanced_sample

        train, _calib = split_train(load_claims(raw_dir / "train.jsonl"))
        sample = balanced_sample(train, per_label, cfg.seed)
        t0 = time.perf_counter()
        examples = make_examples(sample, cfg.index_dir, cfg.seed, workers)
        examples_file.parent.mkdir(parents=True, exist_ok=True)
        with examples_file.open("w", encoding="utf-8") as fh:
            for ex in examples:
                fh.write(json.dumps(dataclasses.asdict(ex)) + "\n")
        typer.echo(f"wrote {len(examples)} examples from {len(sample)} claims to "
                   f"{examples_file} in {time.perf_counter() - t0:.0f}s")  # fmt: skip
    elif stage == "train":
        from witness.retrieval.reranker import resolve_device
        from witness.verify.train import TrainConfig, train_verifier

        with examples_file.open(encoding="utf-8") as fh:
            loaded = [Example(**json.loads(line)) for line in fh]
        stats = train_verifier(
            loaded, out, TrainConfig(epochs=epochs, seed=cfg.seed), resolve_device(cfg.device)
        )
        typer.echo(json.dumps(stats, indent=2))
    else:
        raise typer.BadParameter("--stage must be 'examples' or 'train'")


eval_app = typer.Typer(no_args_is_help=True, help="Benchmarks against published datasets.")
app.add_typer(eval_app, name="eval")


@eval_app.command("fever")
def eval_fever_cmd(
    split: Annotated[str, typer.Option(help="dev (shared_task_dev) or calib")] = "dev",
    limit: Annotated[int, typer.Option(help="Seeded random subset size (0 = all)")] = 0,
    out: Annotated[Path | None, typer.Option(help="Run directory")] = None,
    verifier: Annotated[
        list[str] | None, typer.Option(help="name=model_or_path; first one is the main model")
    ] = None,
    stage: Annotated[
        str, typer.Option(help="Comma-separated subset of candidates,retrieve,verify")
    ] = "candidates,retrieve,verify",
    temperatures: Annotated[Path | None, typer.Option(help="temperatures.json from calib")] = None,
    raw_dir: Annotated[Path, typer.Option()] = Path("data/raw"),
    config: ConfigOpt = None,
) -> None:
    """Label accuracy, FEVER score, evidence recall and ablations on FEVER."""
    from witness.evaluation.commands import run_eval
    from witness.retrieval.reranker import resolve_device

    _setup_logging()
    cfg = load_config(config)
    specs = verifier or [f"finetuned={cfg.verifier.model}"]
    parsed: dict[str, str] = {}
    for spec in specs:
        name, sep, model = spec.partition("=")
        if not sep or not name or not model:
            raise typer.BadParameter(f"--verifier expects name=model, got {spec!r}")
        parsed[name] = model
    results = run_eval(
        cfg,
        split=split,
        raw_dir=raw_dir,
        out_dir=out or Path(f"runs/fever_{split}"),
        limit=limit or None,
        verifiers=parsed,
        main_verifier=next(iter(parsed)),
        stages={x.strip() for x in stage.split(",") if x.strip()},
        temperatures_file=temperatures,
        device=resolve_device(cfg.device),
    )
    typer.echo(json.dumps(results, indent=2))


def main() -> None:
    try:
        app()
    except WitnessError as exc:
        typer.secho(f"error: {exc}", fg=typer.colors.RED, err=True)
        raise SystemExit(2) from exc
