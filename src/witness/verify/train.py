"""Fine-tune an NLI-initialised DeBERTa-v3 cross-encoder as the FEVER verifier.

The NLI head is kept (entailment / contradiction / neutral already mean support /
refute / not-enough-info), so step 0 is the zero-shot model and fine-tuning only adapts
it to multi-sentence Wikipedia evidence. Plain PyTorch loop: bf16 autocast, AdamW,
linear warmup/decay, dynamic padding, fixed seeds.
"""

from __future__ import annotations

import json
import logging
import math
import random
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from witness.verify.train_data import Example
from witness.verify.verifier import LABELS, label_permutation

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class TrainConfig:
    init_model: str = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
    epochs: int = 1
    batch_size: int = 32
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_fraction: float = 0.06
    max_length: int = 320
    seed: int = 13


def train_verifier(
    examples: Sequence[Example], out_dir: Path, config: TrainConfig, device: str
) -> dict[str, float]:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    tokenizer = AutoTokenizer.from_pretrained(config.init_model)
    # Master weights must be fp32: transformers loads a checkpoint in its stored dtype by
    # default, and this one is fp16, which overflows to NaN on the first update.
    model = AutoModelForSequenceClassification.from_pretrained(
        config.init_model, dtype=torch.float32
    ).to(device)
    perm = label_permutation({int(k): str(v) for k, v in model.config.id2label.items()})
    targets = torch.tensor([perm[e.label] for e in examples], dtype=torch.long)

    encodings = tokenizer(
        [e.premise for e in examples],
        [e.claim for e in examples],
        truncation="only_first",
        max_length=config.max_length,
    )
    n = len(examples)
    steps_per_epoch = math.ceil(n / config.batch_size)
    total_steps = steps_per_epoch * config.epochs
    warmup = max(1, int(total_steps * config.warmup_fraction))
    no_decay = ("bias", "LayerNorm.weight", "layernorm", "norm")
    groups = [
        {"params": [p for name, p in model.named_parameters()
                    if not any(nd in name for nd in no_decay)],
         "weight_decay": config.weight_decay},
        {"params": [p for name, p in model.named_parameters()
                    if any(nd in name for nd in no_decay)],
         "weight_decay": 0.0},
    ]  # fmt: skip
    optimizer = torch.optim.AdamW(groups, lr=config.learning_rate)

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        return max(0.0, (total_steps - step) / max(1, total_steps - warmup))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    model.train()
    started = time.perf_counter()
    step = 0
    skipped = 0
    running = 0.0
    on_gpu = device.startswith("cuda")
    generator = torch.Generator().manual_seed(config.seed)
    for _epoch in range(config.epochs):
        order = torch.randperm(n, generator=generator).tolist()
        for start in range(0, n, config.batch_size):
            idx = order[start : start + config.batch_size]
            features = [{k: encodings[k][i] for k in encodings} for i in idx]
            batch = tokenizer.pad(features, return_tensors="pt").to(device)
            labels = targets[idx].to(device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=on_gpu):
                logits = model(**batch).logits
            loss = torch.nn.functional.cross_entropy(logits.float(), labels)
            loss.backward()  # type: ignore[no-untyped-call]  # torch stubs leave it untyped
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            step += 1
            if not (torch.isfinite(loss) and torch.isfinite(norm)):
                # Skip the update rather than poison the weights; give up if it recurs.
                optimizer.zero_grad(set_to_none=True)
                skipped += 1
                if skipped > max(5, total_steps // 50):
                    raise FloatingPointError(f"{skipped} non-finite steps by step {step}")
                continue
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            running += float(loss.detach())
            if step % 50 == 0:
                log.info("step %d/%d loss %.4f (%.0fs)", step, total_steps, running / 50,
                         time.perf_counter() - started)  # fmt: skip
                running = 0.0
    seconds = time.perf_counter() - started
    model.config.id2label = {perm[i]: LABELS[i] for i in range(3)}
    model.config.label2id = {LABELS[i]: perm[i] for i in range(3)}
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    stats = {
        "examples": float(n),
        "steps": float(total_steps),
        "skipped_steps": float(skipped),
        "train_seconds": round(seconds, 1),
        "examples_per_second": round(n * config.epochs / seconds, 1),
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)
        if device.startswith("cuda") else 0.0,
    }  # fmt: skip
    (out_dir / "train_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return stats
