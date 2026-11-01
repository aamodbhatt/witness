"""Verifier training examples from FEVER train, with retrieval-like noise.

At test time the verifier sees the top-5 retrieved sentences, usually a mix of real
evidence and plausible distractors. Training on gold sentences alone would teach it that
every sentence matters, so each example is built to look like retrieval output:

* SUPPORTS / REFUTES: one complete gold evidence group, padded to ``k`` sentences with
  the most claim-like *other* sentences from the gold pages and the linker's pages.
* NOT ENOUGH INFO: the ``k`` most claim-like sentences from the pages the entity linker
  finds for the claim (FEVER provides no evidence for NEI claims).

Sentence order is shuffled so the model cannot learn "evidence comes first".
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from witness.config import LinkerConfig
from witness.evaluation.fever_data import FeverClaim
from witness.retrieval.bm25 import Tokenizer
from witness.retrieval.linker import EntityLinker
from witness.verify.verifier import LABELS
from witness.wiki.store import Page, WikiStore


@dataclass(frozen=True, slots=True)
class Example:
    claim_id: int
    claim: str
    premise: str
    label: int  # index into LABELS


class ExampleBuilder:
    def __init__(self, index_dir: Path, k: int = 5) -> None:
        self.store = WikiStore(index_dir / "wiki.sqlite")
        self.linker = EntityLinker(self.store, np.load(index_dir / "inlinks.npy"), LinkerConfig())
        self.k = k
        self._tokenize = Tokenizer()

    def _ranked(self, claim: str, pages: Sequence[Page], exclude: set[tuple[int, int]]
                ) -> list[tuple[Page, int]]:  # fmt: skip
        query = set(self._tokenize(claim))
        scored: list[tuple[float, int, Page, int]] = []
        for page in pages:
            for line, text in enumerate(page.sentences):
                if text and (page.id, line) not in exclude:
                    terms = self._tokenize(page.title + " " + text)
                    overlap = len(query & set(terms)) / np.sqrt(max(len(terms), 1))
                    scored.append((-overlap, len(scored), page, line))
        scored.sort(key=lambda t: (t[0], t[1]))
        return [(p, ln) for _, _, p, ln in scored]

    def build(self, claim: FeverClaim, seed: int) -> Example | None:
        rng = random.Random(seed * 1_000_003 + claim.id)
        chosen: list[tuple[Page, int]] = []
        linked_pages = self.store.pages([p.page for p in self.linker.link(claim.claim, 4)])
        if claim.label != "NOT ENOUGH INFO":
            group = rng.choice(claim.evidence)
            if len(group) > self.k:
                return None
            ids: list[int] = []
            for title, _ in group:
                page_id = self.store.page_id(title)
                if page_id is None:
                    return None
                ids.append(page_id)
            gold_pages = self.store.pages(ids)
            for page, (_, line) in zip(gold_pages, group, strict=True):
                if line >= len(page.sentences) or not page.sentences[line]:
                    return None
                chosen.append((page, line))
            pool = list({p.id: p for p in gold_pages + linked_pages}.values())
        else:
            pool = linked_pages
        exclude = {(p.id, ln) for p, ln in chosen}
        distractors = self._ranked(claim.claim, pool, exclude)[: self.k * 2]
        rng.shuffle(distractors)
        chosen.extend(distractors[: self.k - len(chosen)])
        if not chosen:
            return None
        rng.shuffle(chosen)
        premise = " ".join(f"{p.title}: {p.sentences[ln]}" for p, ln in chosen)
        return Example(claim.id, claim.claim, premise, LABELS.index(claim.label))


def balanced_sample(claims: Sequence[FeverClaim], per_label: int, seed: int) -> list[FeverClaim]:
    rng = random.Random(seed)
    out: list[FeverClaim] = []
    for label in LABELS:
        pool = [c for c in claims if c.label == label]
        out.extend(rng.sample(pool, min(per_label, len(pool))))
    rng.shuffle(out)
    return out
