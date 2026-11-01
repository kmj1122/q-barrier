"""Finite CPU pipeline checks, not scientific training or evaluation runs.

Use synthetic transitions and temporary checkpoints. Never call Experiment.learn,
interact, or evaluate_val, and never modify the bundled pretrained weights.
"""
from pathlib import Path
from types import SimpleNamespace
from functools import partial

import gymnasium as gym
import numpy as np
import pytest
import torch
import torch.nn.functional as F
from accelerate import Accelerator

from amago.experiment import Experiment
from amago.loading import MAGIC_PAD_VAL, RLData, RLData_pad_collate
from amago.nets.tstep_encoders import FFTstepEncoder
from amago.nets.traj_encoders import TformerTrajEncoder
from amago.shielded_agent import ShieldedAMAGOAgent


@pytest.fixture(autouse=True)
def cpu_only_accelerator(monkeypatch):
    # Keep this software test finite on GPU machines too; do not alter the
    # production Experiment or rely on CUDA_VISIBLE_DEVICES after torch imports.
    monkeypatch.setattr("amago.experiment.Accelerator", partial(Accelerator, cpu=True))


class SmallTstepEncoder(FFTstepEncoder):
    def __init__(self, obs_space, rl2_space):
        super().__init__(obs_space, rl2_space, n_layers=1, d_hidden=32, d_output=32)


class SmallTrajEncoder(TformerTrajEncoder):
    def __init__(self, tstep_dim, max_seq_len):
        super().__init__(
            tstep_dim, max_seq_len, d_model=32, n_heads=4, n_layers=1,
            d_ff=64, dropout_ff=0.0, dropout_emb=0.0,
        )


def make_experiment(tmp_path, continuous=False, critics=2):
    """Use the real model/optimizer initializer, without starting environments."""
    action_space = (
        gym.spaces.Box(-1, 1, shape=(2,), dtype=np.float32)
        if continuous else gym.spaces.Discrete(5)
    )
    action_dim = 2 if continuous else 5
    obs_space = gym.spaces.Dict({
        "observation": gym.spaces.Box(-1, 1, shape=(4,), dtype=np.float32),
        "cost_to_go": gym.spaces.Box(0, 2, shape=(1,), dtype=np.float32),
    })
    constraint = SimpleNamespace(
        fixed_lagrangian_multiplier=1.0, lambda_lr=1e-3, lambda_init=0.0,
        lambda_upper_bound=None, max_unused_fraction=1.0,
        apply_cost_violation_mask=False,
    )
    experiment = Experiment(
        run_name="synthetic-smoke", ckpt_base_dir=str(tmp_path), max_seq_len=8,
        dataset=None, make_train_env=None, make_val_env=None,
        tstep_encoder_type=SmallTstepEncoder, traj_encoder_type=SmallTrajEncoder,
        agent_type=ShieldedAMAGOAgent, val_timesteps_per_epoch=0,
        constrained_config=constraint, batch_size=2, lr_warmup_steps=0,
        verbose=False, log_to_wandb=False, policy_kwargs={
            "num_critics": critics, "num_critics_td": min(critics, 2),
            "use_multigamma": False, "wm_hidden_dim": 32,
            "shield_continuous_num_samples": 4,
            "shield_continuous_enable_fast_path": False,
        },
    )
    experiment.rl2_space = {
        "obs": obs_space,
        "rl2": gym.spaces.Box(-np.inf, np.inf, shape=(action_dim + 1,), dtype=np.float32),
    }
    experiment.train_envs = SimpleNamespace(single_action_space=action_space)
    experiment.init_model()
    assert experiment.DEVICE.type == "cpu", "Run smoke tests with CUDA_VISIBLE_DEVICES=''"
    experiment.init_checkpoints()
    return experiment


def make_batch(continuous=False):
    """Valid replay actions; unequal sequence lengths; one inner reset each."""
    samples = []
    for length in (5, 3):
        actions = (
            torch.linspace(-0.5, 0.5, (length - 1) * 2).reshape(length - 1, 2)
            if continuous else F.one_hot(torch.arange(length - 1) % 5, 5).float()
        )
        soft_resets = torch.zeros(length - 1)
        soft_resets[-1] = 1.0
        samples.append(RLData(
            obs={"observation": torch.rand(length, 4) * 2 - 1,
                 "cost_to_go": torch.ones(length, 1)},
            rews=torch.linspace(0.0, 0.2, length - 1).unsqueeze(-1),
            dones=torch.zeros(length - 1, 1), actions=actions,
            time_idxs=torch.arange(length).unsqueeze(-1),
            infos={"cost": torch.full((length - 1,), 0.01),
                   "cost_limit": torch.ones(length - 1), "soft_reset": soft_resets},
        ))
    return RLData_pad_collate(samples)


@pytest.mark.parametrize("continuous", [False, True])
def test_real_train_step_updates_online_weights_and_masks_padding(tmp_path, continuous):
    torch.manual_seed(42)
    experiment = make_experiment(tmp_path, continuous)
    experiment.policy.train()
    policy = experiment.policy
    batch = make_batch(continuous).to(experiment.DEVICE)
    encoder_before = next(policy.tstep_encoder.parameters()).detach().clone()
    dynamics_before = policy.dyn_z_w1.weight.detach().clone()
    frozen_before = policy.rnd_target_w1.weight.detach().clone()
    target = next(policy.target_critics.parameters())
    online_before = next(policy.critics.parameters()).detach().clone()
    with torch.no_grad():
        target.add_(0.1)
    target_before = target.detach().clone()

    metrics = experiment.train_step(batch, log_step=True)

    assert torch.isfinite(metrics["Actor Loss"])
    assert torch.isfinite(metrics["Critic Loss"])
    assert metrics["Sequence Length"] == 9
    assert metrics["Unmasked Batch Size (in Timesteps)"].item() == 6
    assert batch.rl2s.shape[1] == 9 and batch.actions.shape[1] == 8
    assert torch.all(batch.rl2s[:, 5:] == MAGIC_PAD_VAL)
    assert all(value.shape[1] == 8 for value in batch.infos.values())
    assert experiment.grad_update_counter == 1
    assert experiment.lr_schedule.scheduler.last_epoch >= 1
    assert experiment.optimizer.state
    assert not torch.equal(encoder_before, next(policy.tstep_encoder.parameters()))
    assert not torch.equal(dynamics_before, policy.dyn_z_w1.weight)
    assert torch.equal(frozen_before, policy.rnd_target_w1.weight)
    assert policy.rnd_target_w1.weight.grad is None
    torch.testing.assert_close(
        target, (1 - policy.tau) * target_before + policy.tau * online_before,
    )
    assert all(torch.isfinite(parameter).all() for parameter in policy.parameters())


def eval_inputs(continuous=False, batch_size=2):
    return {
        "obs": {"observation": torch.zeros(batch_size, 1, 4),
                "cost_to_go": torch.ones(batch_size, 1, 1)},
        "rl2s": torch.zeros(batch_size, 1, 3 if continuous else 6),
        "time_idxs": torch.zeros(batch_size, 1, 1, dtype=torch.long),
    }


@pytest.mark.parametrize("continuous", [False, True])
def test_checkpoint_roundtrip_and_cached_context_rollover(tmp_path, continuous):
    torch.manual_seed(42)
    original = make_experiment(tmp_path / "original", continuous)
    original.save_policy_weights_checkpoint()
    checkpoint = Path(original.ckpt_dir) / "policy_weights/policy_epoch_0.pt"
    restored = make_experiment(tmp_path / "restored", continuous)
    restored.load_checkpoint_from_path(str(checkpoint), is_accelerate_state=False)
    original.policy.eval()
    restored.policy.eval()
    for key, value in original.policy.state_dict().items():
        assert torch.equal(value, restored.policy.state_dict()[key]), key
    inputs = eval_inputs(continuous)
    with torch.no_grad():
        torch.manual_seed(123)
        first, _ = original.policy.get_actions(**inputs, sample=False)
        torch.manual_seed(123)
        second, _ = restored.policy.get_actions(**inputs, sample=False)
    assert torch.equal(first, second)

    encoder = restored.policy.traj_encoder
    cache = encoder.init_hidden_state(batch_size=2, device=restored.DEVICE)
    with torch.no_grad():
        for step in range(restored.max_seq_len + 3):
            inputs["time_idxs"].fill_(step)
            actions, cache = restored.policy.get_actions(**inputs, hidden_state=cache, sample=False)
            assert actions.shape == (2, 1, 2 if continuous else 1)
            assert torch.isfinite(actions).all()
            expected_length = min(step + 1, restored.max_seq_len - 1)
            assert torch.all(cache.seq_lens == expected_length)
            for index in range(2):
                valid = int(cache.seq_lens[index])
                assert torch.isfinite(cache.key_cache.data[:, index, :valid]).all()
                assert torch.isfinite(cache.val_cache.data[:, index, :valid]).all()
    assert cache.key_cache.data.shape[2] == restored.max_seq_len
    retained = int(cache.seq_lens[1])
    assert retained > 0
    encoder.reset_hidden_state(cache, np.array([True, False]))
    assert int(cache.seq_lens[0]) == 0
    assert int(cache.seq_lens[1]) == retained
    # Reset one batch slot: the next latent must agree with an empty-cache slot.
    seq = torch.randn(2, 1, encoder.tstep_dim)
    times = torch.full((2, 1, 1), 13, dtype=torch.long)
    fresh = encoder.init_hidden_state(batch_size=2, device=restored.DEVICE)
    with torch.no_grad():
        actual, _ = encoder(seq, times, cache)
        expected, _ = encoder(seq, times, fresh)
    torch.testing.assert_close(actual[0], expected[0], atol=1e-5, rtol=1e-5)


def test_checkpoint_missing_backbone_key_is_rejected(tmp_path):
    experiment = make_experiment(tmp_path)
    weights = experiment.policy.state_dict()
    key = next(key for key in weights if key.startswith("tstep_encoder."))
    del weights[key]
    path = tmp_path / "incomplete.pt"
    torch.save(weights, path)
    with pytest.raises(RuntimeError, match="Missing key"):
        experiment.load_checkpoint_from_path(str(path), is_accelerate_state=False)


def test_checkpoint_incompatible_critic_count_is_rejected(tmp_path):
    original = make_experiment(tmp_path / "original")
    original.save_policy_weights_checkpoint()
    incompatible = make_experiment(tmp_path / "incompatible", critics=1)
    path = Path(original.ckpt_dir) / "policy_weights/policy_epoch_0.pt"
    with pytest.raises(RuntimeError, match="size mismatch"):
        incompatible.load_checkpoint_from_path(str(path), is_accelerate_state=False)


@pytest.mark.parametrize("fast_init", [False, True])
def test_sigma_buffers_have_compact_storage_for_safetensors(monkeypatch, fast_init):
    from amago.nets.transformer import SigmaReparam

    # Exercise SVD initialization even on hosts that normally use fast-init.
    monkeypatch.setattr("amago.nets.transformer._cpu_supports_avx512", lambda: True)
    layer = SigmaReparam(8, 24, fast_init=fast_init)
    for buffer in (layer.u, layer.v):
        assert buffer.is_contiguous()
        assert buffer.storage_offset() == 0
        assert buffer.untyped_storage().nbytes() == buffer.numel() * buffer.element_size()


@pytest.mark.parametrize("continuous", [False, True])
def test_full_training_state_restores_optimizer_scheduler_and_rng(tmp_path, monkeypatch, continuous):
    import random

    monkeypatch.setattr("amago.nets.transformer._cpu_supports_avx512", lambda: True)
    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)
    original = make_experiment(tmp_path / "original", continuous)
    original.train_step(make_batch(continuous).to(original.DEVICE), log_step=True)
    original.save_checkpoint()
    state_path = Path(original.ckpt_dir) / "training_states/synthetic-smoke_epoch_0"
    expected_python = random.random()
    expected_numpy = np.random.random()
    expected_torch = torch.rand(4)
    expected_optimizer = original.optimizer.state_dict()
    expected_scheduler = original.lr_schedule.state_dict()

    restored = make_experiment(tmp_path / "restored", continuous)
    safe_globals_before = set(torch.serialization.get_safe_globals())
    restored.load_checkpoint_from_path(str(state_path), is_accelerate_state=True)
    assert set(torch.serialization.get_safe_globals()) == safe_globals_before
    assert random.random() == expected_python
    assert np.random.random() == expected_numpy
    torch.testing.assert_close(torch.rand(4), expected_torch, atol=0, rtol=0)
    for name, value in original.policy.state_dict().items():
        assert torch.equal(value, restored.policy.state_dict()[name]), name
    actual_optimizer = restored.optimizer.state_dict()
    assert actual_optimizer["param_groups"] == expected_optimizer["param_groups"]
    assert actual_optimizer["state"].keys() == expected_optimizer["state"].keys()
    for parameter_id, values in expected_optimizer["state"].items():
        for name, value in values.items():
            actual = actual_optimizer["state"][parameter_id][name]
            if isinstance(value, torch.Tensor):
                assert torch.equal(value, actual), name
            else:
                assert value == actual
    assert restored.lr_schedule.state_dict() == expected_scheduler

    before = restored.policy.dyn_z_w1.weight.detach().clone()
    metrics = restored.train_step(make_batch(continuous).to(restored.DEVICE), log_step=True)
    assert torch.isfinite(metrics["Actor Loss"]) and torch.isfinite(metrics["Critic Loss"])
    assert not torch.equal(before, restored.policy.dyn_z_w1.weight)
    assert {int(v["step"]) for v in restored.optimizer.state_dict()["state"].values()} == {2}


@pytest.mark.parametrize("preexisting", [False, True])
@pytest.mark.parametrize("load_fails", [False, True])
def test_legacy_rng_dtype_scope_preserves_caller_allowlist(monkeypatch, preexisting, load_fails):
    import amago.experiment as experiment_module

    dtype = type(np.dtype(np.uint32))
    initial = set(torch.serialization.get_safe_globals())
    # Simulate the legacy API locally without changing NumPy for other modules.
    monkeypatch.setattr(experiment_module, "np", SimpleNamespace(dtype=np.dtype, uint32=np.uint32))

    def load_state(path):
        assert path == "unused-by-observer"
        assert dtype in torch.serialization.get_safe_globals()
        if load_fails:
            raise RuntimeError("deliberate load failure")

    observed = SimpleNamespace(accelerator=SimpleNamespace(print=lambda *args: None, load_state=load_state))
    additions = [dtype] if preexisting and dtype not in initial else []
    with torch.serialization.safe_globals(additions):
        before = set(torch.serialization.get_safe_globals())
        if load_fails:
            with pytest.raises(RuntimeError, match="deliberate load failure"):
                Experiment.load_checkpoint_from_path(observed, "unused-by-observer", is_accelerate_state=True)
        else:
            Experiment.load_checkpoint_from_path(observed, "unused-by-observer", is_accelerate_state=True)
        assert set(torch.serialization.get_safe_globals()) == before
    assert set(torch.serialization.get_safe_globals()) == initial
