from pathlib import Path

import pytest

from utils.checkpoints import require_evaluation_checkpoint


def test_no_checkpoint_is_not_silently_accepted():
    with pytest.raises(ValueError, match="requires a trained checkpoint"):
        require_evaluation_checkpoint({"load_checkpoint_path": None, "load_checkpoint_epoch": None})


def test_missing_explicit_checkpoint_fails(tmp_path):
    with pytest.raises(FileNotFoundError, match="checkpoint not found"):
        require_evaluation_checkpoint({"load_checkpoint_path": str(tmp_path / "absent.pt")})


def test_directory_is_not_a_checkpoint(tmp_path):
    with pytest.raises(FileNotFoundError):
        require_evaluation_checkpoint({"load_checkpoint_path": str(tmp_path)})


def test_explicit_path_resolves_without_deserializing(tmp_path):
    path = tmp_path / "model.pt"
    path.write_bytes(b"fixture: existence check only, not model data")
    assert Path(require_evaluation_checkpoint({"load_checkpoint_path": str(path)})) == path.resolve()


def test_original_run_epoch_layout_still_supported(tmp_path):
    path = tmp_path / "run" / "ckpts" / "policy_weights" / "policy_epoch_12.pt"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"fixture")
    config = {"ckpt_base_dir": str(tmp_path), "run_name": "run", "load_checkpoint_epoch": 12}
    assert Path(require_evaluation_checkpoint(config)) == path.resolve()
