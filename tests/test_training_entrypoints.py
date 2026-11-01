"""Mode and checkpoint-routing checks; never start environments or learning loops."""

import importlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
from omegaconf import OmegaConf


class Config(SimpleNamespace):
    def get(self, key, default=None):
        return getattr(self, key, default)


def import_entrypoint(module_name):
    # CLI entrypoints normally run in separate processes. Avoid duplicate global
    # resolver registration when importing both solely to test their helpers.
    with patch.object(OmegaConf, "register_new_resolver"):
        return importlib.import_module(module_name)


@pytest.mark.parametrize("module_name", ["crl2_darkroom", "crl2_mujoco"])
@pytest.mark.parametrize(
    "flags, expected",
    [
        ({}, None),
        ({"collect_only": False, "learn_only": False}, None),
        ({"collect_only": True}, "collect"),
        ({"collect_only": False, "learn_only": True}, "learn"),
        ({"collect_only": True, "learn_only": True}, "collect"),
    ],
)
def test_training_modes_respect_boolean_values(monkeypatch, module_name, flags, expected):
    module = import_entrypoint(module_name)
    experiment = object()
    collect = Mock(return_value=object())
    learn = Mock(return_value=object())
    monkeypatch.setattr(module, "make_experiment_collect_only", collect)
    monkeypatch.setattr(module, "make_experiment_learn_only", learn)
    result = module.configure_training_mode(experiment, Config(**flags))
    if expected == "collect":
        assert result is collect.return_value
        collect.assert_called_once_with(experiment)
        learn.assert_not_called()
    elif expected == "learn":
        assert result is learn.return_value
        learn.assert_called_once_with(experiment)
        collect.assert_not_called()
    else:
        assert result is experiment
        collect.assert_not_called()
        learn.assert_not_called()


@pytest.mark.parametrize("resume_state", [True, False])
@pytest.mark.parametrize("redirect", ["run", "base", "both"])
def test_redirected_resume_uses_saved_layout(tmp_path, resume_state, redirect):
    module = import_entrypoint("crl2_mujoco")
    config = Config(
        run_name="destination", ckpt_base_dir=str(tmp_path / "dest"),
        resume_from_epoch=12, resume_training_state=resume_state,
    )
    if redirect in ("run", "both"):
        config.load_run_name = "source"
    if redirect in ("base", "both"):
        config.load_ckpt_base_dir = str(tmp_path / "source")
    source_name = config.get("load_run_name", config.run_name)
    source_base = Path(config.get("load_ckpt_base_dir", config.ckpt_base_dir))
    root = source_base / source_name / "ckpts"
    expected = (
        root / "training_states" / f"{source_name}_epoch_12"
        if resume_state else root / "policy_weights" / "policy_epoch_12.pt"
    )
    experiment = Mock(epoch=0)
    module.resume_training(experiment, config)
    experiment.load_checkpoint_from_path.assert_called_once_with(
        str(expected), is_accelerate_state=resume_state,
    )
    experiment.load_checkpoint.assert_not_called()
    assert experiment.epoch == 12


@pytest.mark.parametrize("resume_state", [True, False])
def test_same_run_resume_keeps_experiment_helper(resume_state):
    module = import_entrypoint("crl2_mujoco")
    config = Config(
        run_name="run", ckpt_base_dir="checkpoints",
        resume_from_epoch=12, resume_training_state=resume_state,
    )
    experiment = Mock()
    module.resume_training(experiment, config)
    experiment.load_checkpoint.assert_called_once_with(
        epoch=12, resume_training_state=resume_state,
    )
    experiment.load_checkpoint_from_path.assert_not_called()
