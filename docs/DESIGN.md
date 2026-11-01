# witness: design

## Goals

1. Put a verdict under every check-worthy sentence of any web page, in place, with the
   evidence one hover away. Green: Wikipedia supports it. Red: Wikipedia contradicts it.
   Grey: it cannot tell.
2. Run entirely on the user's machine: no cloud API, no telemetry.
3. Be measurable: every number in the README comes from a script in this repository run
   against a public benchmark (FEVER), with the published baselines next to it.

## Non-goals

- **Being an oracle.** The knowledge source is a June 2017 Wikipedia snapshot. Anything
  newer, or anything Wikipedia's lead sections do not state, should come out grey.
- **Judging opinions.** Questions, first-person statements and evaluative language are
  skipped before verification, with a reason.
- **Other languages.** English only (tokenizer, stopwords, check-worthiness rules).

## Pipeline

```
page paragraphs ─▶ sentence spans ─▶ check-worthiness ─▶ decontextualise ─┐
                                                                          ▼
 verdict ◀─ abstention ◀─ temperature ◀─ DeBERTa-v3 verifier ◀─ top-5 sentences
                                                                          ▲
     entity linker (titles, disambiguated titles, anchors) ─┐             │
                                                            ├─ RRF ─▶ cross-encoder
     BM25 over 5.4M pages (title + lead) ───────────────────┘
```

| Stage | Module | Why it exists |
|---|---|---|
| Sentence spans | `claims/segment.py` | The browser underlines exact character ranges, so offsets must be reproducible. |
| Check-worthiness | `claims/extract.py` | Opinions, questions and boilerplate would otherwise be painted grey and drown the signal. |
| Decontextualisation | `claims/extract.py` | "She was born in Warsaw" is uncheckable without "Marie Curie"; only the query changes, the underline stays on the original words. |
| Entity linker | `retrieval/linker.py` | FEVER claims are about named entities; a title/alias lookup finds the right page far more often than any search (see ablations). |
| BM25 | `retrieval/bm25.py` | Catches pages the claim describes without naming (paraphrased titles, misspellings, descriptive claims). |
| Fusion | `retrieval/pipeline.py` | Weighted reciprocal-rank fusion needs no score calibration between a lookup and a search engine. |
| Cross-encoder | `retrieval/reranker.py` | Picks the 5 sentences (out of ~100 candidates) the verifier will read; the biggest single lever on FEVER score. |
| Verifier | `verify/verifier.py`, `verify/train.py` | Three-way NLI over the concatenated evidence. |
| Temperature + abstention | `verify/calibration.py`, `verify/verdict.py` | The hover card prints a confidence, so it has to mean something; low-confidence calls turn grey instead of green/red. |

## Data model and invariants

- **Page ids** are dense integers `0..N-1`, assigned in dump order. They are the SQLite
  primary key *and* the BM25 matrix row, so a search hit is one primary-key lookup away
  from its sentences.
- **Sentence numbers are FEVER's.** The dump's `lines` field leaves empty slots between
  paragraphs; the store keeps them as empty strings, so `(page, line)` in our output is
  exactly what the official scorer compares against. A unit test pins this.
- **Text is detokenised once, at build time** (`-LRB-` → `(`, ` ,` → `,`), because the
  models were pre-trained on natural text and the hover card shows it to people.
- **Claim offsets are into the exact string the browser sent** (`element.textContent`),
  so the content script can map them back to DOM text nodes without re-segmenting.
- **Underlines never change page text.** The content script splits text nodes and wraps
  pieces in spans; links and inline formatting inside a claim survive. The browser test
  asserts the paragraph's text is unchanged.

## Indexing within a 50 GB RAM pod (and a laptop)

`bm25s` builds its score matrix from the whole tokenized corpus held as Python lists,
which for 5.4M pages (212M postings) needs far more memory than the 12 GB per-process
budget. `wiki/build.py` instead:

1. parses the 109 dump shards in a 7-process pool, each returning *per-document term
   counts* with a shard-local vocabulary (compact lists, cheap to pickle);
2. remaps them to a global vocabulary and appends to numpy arrays (~2 GB);
3. computes Lucene-variant BM25 weights in one vectorised pass and lets scipy convert
   CSR → CSC (term-major) in C++.

The result is three `.npy` files loaded with `mmap_mode="r"`: a query touches only the
posting lists of its terms. `tests/test_bm25.py` checks scores against
`bm25s.BM25(method="lucene")` to 1e-5. Anchor-text aggregation (~25M raw rows) is a
`GROUP BY` inside SQLite rather than a Python dict.

The **lite profile** (`witness build-index --lite 500000`) is for laptops. It ranks titles
by incoming links (counted from the dump's own hyperlinks) and keeps the top 500k; ties at
the cutoff go to the alphabetically first title, so every build keeps the same pages. About
99k of those titles are link targets with no page in the dump, so the lite index holds
400,856 pages (0.77 GB) and builds in 3 to 7 minutes on 8 cores, link counting included.

`configs/laptop.toml` pairs the lite index with the off-the-shelf
`MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` verifier, because the fine-tuned weights
(0.75 GB) are not distributed and training needs a CUDA GPU. Its temperature (T = 2.43,
`configs/calibration/mnli-fever-anli.json`) was fit on the same 2,000 calibration claims
as the fine-tuned model's. With the full index that verifier scores 71.70 label accuracy
and 65.17 FEVER score on dev; the lite-index combination has not been scored. On the demo
article it marks all three altered facts red, plus two true sentences.

## Evaluation methodology

- **Splits.** `train.jsonl` is split by a hash of the claim id into *train* (98%) and a
  *calibration bucket* (2%). The verifier is trained only on a balanced sample of the
  train part; the temperature is fit only on 2,000 calibration claims run through the
  full pipeline; `shared_task_dev.jsonl` (19,998 claims) is only scored. No threshold,
  weight or temperature is chosen on dev.
- **Scorer.** `evaluation/fever_scorer.py` reimplements the official `fever-scorer`
  (strict FEVER score: right label *and* a complete gold evidence group within the
  first five predicted sentences). Its tests include the official README example.
- **Ablations share candidates.** One pass computes linker and BM25 candidates and
  cross-encoder scores for the union of their pages; each ablation then selects from
  the same cached scores. Differences between rows are caused by the ablated component
  only.
- **Off-the-shelf baselines** run over the *same* retrieved evidence. Note that
  `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` (also our initialisation) saw
  FEVER-NLI training data derived from FEVER *train*; `cross-encoder/nli-deberta-v3-base`
  (SNLI + MNLI) is the strictly zero-shot comparison.
- **Published numbers** are copied from the KGAT paper (Liu et al., ACL 2020, Table 2)
  and labelled as reported. They use other retrieval stacks and, for KGAT, BERT/RoBERTa
  verifiers trained on the full FEVER train set.

### Leakage checks

- Verifier training examples are built from FEVER train claims only; the calibration
  bucket is excluded by construction (`fever_data.split_train`).
- The entity linker, BM25 and the alias table are built from the Wikipedia dump alone
  and never see claims or labels.
- The cross-encoder (`ms-marco-MiniLM-L6-v2`) is used off-the-shelf; it was trained on
  MS MARCO, not FEVER.

## Verifier training

- Initialised from an NLI model whose head already means entail / neutral / contradict,
  so step 0 is a sensible zero-shot verifier and fine-tuning only adapts it to
  multi-sentence Wikipedia evidence and FEVER's label semantics.
- Inputs look like test-time retrieval: a complete gold evidence group padded to five
  sentences with the most claim-like *other* sentences from the gold pages and the
  linker's pages, in shuffled order; NOT ENOUGH INFO claims get the five most
  claim-like sentences from their linked pages.
- Balanced sample (17k claims per label from the train split), one epoch, AdamW 2e-5,
  linear warmup/decay, bf16 autocast, max length 320, seed 13.

## Failure modes

| Failure | Behaviour |
|---|---|
| Server not running | Popup and options page say "offline"; nothing is underlined. |
| Index missing or stale | `IndexStoreError` naming the missing file and the command that builds it; `train-verifier` checks before it starts worker processes. |
| No trained verifier (a fresh clone has no `artifacts/verifier`) | `BackendError` before the server binds its port, naming the model and both fixes: train one, or use a Hub NLI model as `configs/laptop.toml` does. |
| Claim about post-2017 facts | Usually grey; can be red when the 2017 lead states an older fact (e.g. a former office holder). |
| Entity not in the snapshot | Grey: no evidence sentence is relevant enough for a confident call. |
| Pronoun chains across paragraphs | Decontextualisation resets the subject at each paragraph to the page title; a wrong subject produces a wrong query, visible as "Checked as: …" in the hover card. |
| Very long pages | At most 120 blocks and 60 claims per page are checked (configurable). |

## Alternatives considered

- **Dense retrieval** (binary-quantised first stage + int8 rescoring). Designed but not
  shipped in this version: the time and GPU budget went to the verifier. The linker
  already reaches ~86% document recall on FEVER dev, so the expected gain is smaller
  than on paraphrase-heavy benchmarks. It is first on the roadmap.
- **Multi-hop retrieval for HoVer.** Not in this version; FEVER is single-hop for ~85% of
  claims. Following the dump's hyperlinks from hop-1 evidence is the planned approach.
- **An LLM for decontextualisation.** Rules cover the dominant pattern (sentence-initial
  pronoun or "The film"); a small local LLM is a roadmap item.
- **A learned check-worthiness classifier.** Rules with explicit reasons are auditable
  and shown in the API response; a classifier needs labelled web-page data we do not
  have.
