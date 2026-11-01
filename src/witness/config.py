"""Typed configuration, loaded from TOML and validated at the boundary.

Every tunable number lives here. ``configs/pod.toml`` (full 5.4M-page index, GPU),
``configs/cpu.toml`` (full index, CPU) and ``configs/laptop.toml`` (lite index, CPU,
off-the-shelf verifier) are partial overrides of these defaults.
"""

from __future__ import annotations

import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from witness.errors import ConfigError

Device = Literal["auto", "cpu", "cuda"]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class LinkerConfig(_Strict):
    """Entity linking against page titles, disambiguated titles and anchor texts."""

    max_ngram: int = Field(default=8, ge=1, le=12)
    pages_per_mention: int = Field(default=4, ge=1)
    top_k: int = Field(default=8, ge=1)


class BM25Config(_Strict):
    top_k: int = Field(default=8, ge=1)


class FusionConfig(_Strict):
    """Weighted reciprocal-rank fusion of the linker and BM25 page rankings."""

    rrf_k: int = Field(default=10, ge=1)
    linker_weight: float = Field(default=1.0, ge=0.0)
    bm25_weight: float = Field(default=1.0, ge=0.0)
    top_pages: int = Field(default=10, ge=1)

    @model_validator(mode="after")
    def _check(self) -> Self:
        if self.linker_weight == 0 and self.bm25_weight == 0:
            raise ValueError("fusion needs at least one retriever with a positive weight")
        return self


class RerankerConfig(_Strict):
    model: str = "cross-encoder/ms-marco-MiniLM-L6-v2"
    batch_size: int = Field(default=128, ge=1)
    max_length: int = Field(default=128, ge=32)
    max_sentences_per_page: int = Field(default=40, ge=1)
    evidence_k: int = Field(default=5, ge=1, le=10)


class VerifierConfig(_Strict):
    model: str = "artifacts/verifier"
    batch_size: int = Field(default=32, ge=1)
    max_length: int = Field(default=384, ge=64)
    # Temperature fit on held-out FEVER train claims. For a local model folder,
    # `witness eval fever --split calib` writes it there; no file means T = 1.
    calibration_file: Path = Path("artifacts/verifier/calibration.json")
    # Calibrated confidence below this becomes "unverifiable" (grey) in the UI.
    abstain_below: float = Field(default=0.6, ge=0.0, le=1.0)


class ClaimsConfig(_Strict):
    min_words: int = Field(default=5, ge=1)
    max_words: int = Field(default=60, ge=5)
    max_claims_per_page: int = Field(default=60, ge=1)


class ServerConfig(_Strict):
    host: str = "127.0.0.1"
    port: int = Field(default=8094, ge=1, le=65535)


class WitnessConfig(_Strict):
    index_dir: Path = Path("data/index/full")
    device: Device = "auto"
    seed: int = 13
    linker: LinkerConfig = LinkerConfig()
    bm25: BM25Config = BM25Config()
    fusion: FusionConfig = FusionConfig()
    reranker: RerankerConfig = RerankerConfig()
    verifier: VerifierConfig = VerifierConfig()
    claims: ClaimsConfig = ClaimsConfig()
    server: ServerConfig = ServerConfig()

    @classmethod
    def from_toml(cls, path: Path) -> WitnessConfig:
        return cls.from_mapping(load_toml(path))

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> WitnessConfig:
        try:
            return cls.model_validate(dict(data))
        except ValueError as exc:
            raise ConfigError(f"invalid configuration: {exc}") from exc

    def with_overrides(self, overrides: Mapping[str, Any]) -> WitnessConfig:
        """Deep-merge a partial mapping into this config and re-validate."""
        merged = deep_merge(self.model_dump(mode="python"), overrides)
        return WitnessConfig.from_mapping(merged)


def load_toml(path: Path) -> dict[str, Any]:
    try:
        with path.open("rb") as fh:
            return tomllib.load(fh)
    except FileNotFoundError as exc:
        raise ConfigError(f"config file not found: {path}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"config file {path} is not valid TOML: {exc}") from exc


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = dict(base)
    for key, value in override.items():
        current = out.get(key)
        if isinstance(current, Mapping) and isinstance(value, Mapping):
            out[key] = deep_merge(current, value)
        else:
            out[key] = value
    return out


def load_config(path: Path | None) -> WitnessConfig:
    return WitnessConfig() if path is None else WitnessConfig.from_toml(path)
