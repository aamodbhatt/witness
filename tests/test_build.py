"""Index build on a two-shard miniature of the FEVER dump, and the lite page selection."""

from __future__ import annotations

import json
import random
import zipfile
from pathlib import Path

import numpy as np
from hypothesis import given
from hypothesis import strategies as st

from witness.config import LinkerConfig
from witness.retrieval.bm25 import BM25Index
from witness.retrieval.linker import EntityLinker
from witness.wiki.build import build_index, top_titles
from witness.wiki.store import WikiStore


def _page(raw: str, *lines: str) -> str:
    text = " ".join(line.split("\t")[0] for line in lines)
    field = "\n".join(f"{i}\t{line}" for i, line in enumerate(lines))
    return json.dumps({"id": raw, "text": text, "lines": field})


def _dump(path: Path) -> Path:
    shards = {
        "wiki-pages/wiki-001.jsonl": [
            json.dumps({"id": "", "text": "", "lines": ""}),
            _page("Soul_Food_-LRB-film-RRB-",
                  "Soul Food is a 1997 American comedy-drama film .\tcomedy-drama\tComedy-drama",
                  "It was produced by Fox 2000 Pictures .\tFox 2000 Pictures\tFox 2000 Pictures"),
            _page("Fox_2000_Pictures", "Fox 2000 Pictures is an American film studio ."),
        ],
        "wiki-pages/wiki-002.jsonl": [
            _page("Comedy-drama", "Comedy-drama is a genre of film ."),
            _page("Soul_Food", "Soul Food is a dish .\tdish\tSoul Food -LRB-film-RRB-"),
        ],
    }  # fmt: skip
    with zipfile.ZipFile(path, "w") as zf:
        for name, lines in shards.items():
            zf.writestr(name, "\n".join(lines) + "\n")
    return path


def test_build_produces_store_aliases_and_bm25(tmp_path: Path) -> None:
    out = tmp_path / "index"
    stats = build_index(_dump(tmp_path / "wiki-pages.zip"), out, workers=1)
    assert stats["pages"] == 4

    store = WikiStore(out / "wiki.sqlite")
    film = store.page_id("Soul_Food_-LRB-film-RRB-")
    assert film is not None
    page = store.page(film)
    assert page.title == "Soul Food (film)"
    assert page.sentences[1] == "It was produced by Fox 2000 Pictures."
    assert page.links[1] == ("Fox 2000 Pictures",)

    inlinks = np.load(out / "inlinks.npy")
    assert inlinks[store.page_id("Fox_2000_Pictures") or 0] == 1

    linker = EntityLinker(store, inlinks, LinkerConfig())
    linked = [store.page(p.page).raw_title for p in linker.link("Soul Food is a film.")]
    assert linked[0] in {"Soul_Food_-LRB-film-RRB-", "Soul_Food"}
    assert "Soul_Food_-LRB-film-RRB-" in linked

    bm25 = BM25Index.load(out / "bm25")
    top_doc, _ = bm25.search("film studio", 1)[0]
    assert store.page(top_doc).raw_title == "Fox_2000_Pictures"


def test_lite_cutoff_breaks_ties_by_title() -> None:
    counts = {"Zeta": 2, "Beta": 2, "Alpha": 5, "Gamma": 2}
    assert top_titles(counts, 2) == {"Alpha", "Beta"}
    assert top_titles(counts, 10) == set(counts)


@given(
    st.dictionaries(st.text(min_size=1, max_size=4), st.integers(0, 3), max_size=30),
    st.integers(0, 30),
    st.randoms(),
)
def test_lite_selection_ignores_insertion_order(
    counts: dict[str, int], n: int, rng: random.Random
) -> None:
    items = list(counts.items())
    rng.shuffle(items)
    assert top_titles(dict(items), n) == top_titles(counts, n)
    assert len(top_titles(counts, n)) == min(n, len(counts))
