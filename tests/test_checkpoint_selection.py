from pathlib import Path

import pytest

from mpd.inference.inference import resolve_model_checkpoint_path


def _checkpoint(model_dir: Path, filename: str) -> Path:
    path = model_dir / "checkpoints" / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
    return path


def test_defaults_to_ema_current_checkpoint(tmp_path):
    expected = _checkpoint(tmp_path, "ema_model_current.pth")

    assert resolve_model_checkpoint_path(tmp_path, use_ema=True) == expected


def test_defaults_to_non_ema_current_checkpoint(tmp_path):
    expected = _checkpoint(tmp_path, "model_current.pth")

    assert resolve_model_checkpoint_path(tmp_path, use_ema=False, checkpoint="") == expected


def test_selects_intermediate_full_model_checkpoint(tmp_path):
    expected = _checkpoint(tmp_path, "ema_model__iter_500000.pth")

    assert (
        resolve_model_checkpoint_path(
            tmp_path,
            use_ema=True,
            checkpoint="ema_model__iter_500000.pth",
        )
        == expected
    )


@pytest.mark.parametrize("checkpoint", ["../model.pth", "/tmp/model.pth", "checkpoints/model.pth"])
def test_rejects_checkpoint_paths(tmp_path, checkpoint):
    with pytest.raises(ValueError, match="must be a filename"):
        resolve_model_checkpoint_path(tmp_path, use_ema=True, checkpoint=checkpoint)


def test_rejects_state_dict_checkpoint(tmp_path):
    with pytest.raises(ValueError, match="full-model"):
        resolve_model_checkpoint_path(
            tmp_path,
            use_ema=True,
            checkpoint="ema_model__iter_500000_state_dict.pth",
        )


def test_rejects_missing_checkpoint(tmp_path):
    with pytest.raises(FileNotFoundError, match="Model checkpoint not found"):
        resolve_model_checkpoint_path(
            tmp_path,
            use_ema=True,
            checkpoint="ema_model__iter_500000.pth",
        )
