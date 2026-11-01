"""The shipped configs parse, and the laptop profile needs nothing that a clone lacks."""

from __future__ import annotations

from pathlib import Path

import pytest

from witness.config import WitnessConfig
from witness.verify.calibration import load_temperature
from witness.verify.verifier import OFF_THE_SHELF_VERIFIER

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = sorted((ROOT / "configs").glob("*.toml"))


def test_configs_are_found() -> None:
    assert {p.name for p in CONFIGS} >= {"pod.toml", "cpu.toml", "laptop.toml"}


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name)
def test_shipped_config_validates(path: Path) -> None:
    cfg = WitnessConfig.from_toml(path)
    assert not cfg.index_dir.is_absolute()
    assert not Path(cfg.verifier.model).is_absolute()


def test_laptop_profile_runs_without_a_trained_verifier() -> None:
    cfg = WitnessConfig.from_toml(ROOT / "configs" / "laptop.toml")
    assert cfg.device == "cpu"
    assert cfg.index_dir == Path("data/index/lite")
    # A Hub model, not the gitignored artifacts/ folder that only training creates.
    assert cfg.verifier.model == OFF_THE_SHELF_VERIFIER
    calibration = ROOT / cfg.verifier.calibration_file
    assert calibration.is_file(), "the laptop profile's temperature must ship with the repo"
    assert load_temperature(calibration) == pytest.approx(2.4346, abs=1e-4)
