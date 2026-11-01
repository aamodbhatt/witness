"""What a fresh clone hits first: no trained verifier, no index. Both must say what to do."""

from __future__ import annotations

from pathlib import Path

import pytest
import transformers

from witness.errors import BackendError, IndexStoreError
from witness.evaluation.commands import make_examples
from witness.verify.verifier import OFF_THE_SHELF_VERIFIER, NLIVerifier

_HUB_ERROR = (
    "artifacts/verifier is not a local folder and is not a valid model identifier listed "
    "on 'https://huggingface.co/models'\nIf this is a private repository, ..."
)


class _MissingModel:
    @classmethod
    def from_pretrained(cls, *_args: object, **_kwargs: object) -> object:
        raise OSError(_HUB_ERROR)


def test_missing_verifier_names_both_fixes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(transformers, "AutoTokenizer", _MissingModel)
    with pytest.raises(BackendError) as info:
        NLIVerifier("artifacts/verifier", "cpu", max_length=64, batch_size=1)
    message = str(info.value)
    assert "'artifacts/verifier'" in message
    assert "is not a local folder" in message
    assert "If this is a private repository" not in message  # first line of the cause only
    assert "witness train-verifier" in message
    assert "configs/laptop.toml" in message
    assert OFF_THE_SHELF_VERIFIER in message


def test_training_examples_without_an_index_fail_fast(tmp_path: Path) -> None:
    index_dir = tmp_path / "index" / "full"
    with pytest.raises(IndexStoreError, match="witness build-index --out"):
        make_examples([], index_dir, seed=13, workers=1)
