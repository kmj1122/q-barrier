import torch
import torch.nn as nn
import gymnasium as gym
import numpy as np
from amago.agent import Agent
from amago.shielded_agent import ShieldedAMAGOAgent
from amago.nets.tstep_encoders import FFTstepEncoder
from amago.nets.traj_encoders import TformerTrajEncoder
from amago.loading import Batch, MAGIC_PAD_VAL

def test_shielded_agent_full():
    print("=== Testing ShieldedAMAGOAgent: Full Verification ===")
    
    # 1. Define minimal spaces
    obs_space = gym.spaces.Dict({
        "observation": gym.spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32),
        "cost_to_go": gym.spaces.Box(low=0, high=10, shape=(1,), dtype=np.float32)
    })
    # rl2_space: [prev_action (5) + prev_reward (1)] = 6
    rl2_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)
    action_space = gym.spaces.Discrete(5)
    
    class MockConstrainedConfig:
        def __init__(self):
            self.fixed_lagrangian_multiplier = 1.0
            self.lambda_lr = 1e-3
            self.lambda_init = 0.0
            self.lambda_upper_bound = None
            self.max_unused_fraction = 1.0
            self.apply_cost_violation_mask = False
            
    constrained_config = MockConstrainedConfig()
    
    # 2. Instantiate Agent
    max_seq_len = 10
    agent = ShieldedAMAGOAgent(
        obs_space=obs_space,
        rl2_space=rl2_space,
        action_space=action_space,
        max_seq_len=max_seq_len,
        tstep_encoder_type=FFTstepEncoder,
        traj_encoder_type=TformerTrajEncoder,
        constrained_config=constrained_config,
        num_critics=2,
        use_world_model_aux=True,
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent.to(device)
    
    # --- TEST 1: INFERENCE & SHIELD LOGIC ---
    print("\n--- Test 1: get_actions (Inference & Shielding) ---")
    agent.eval()
    batch_size, seq_len = 2, 4
    obs = {
        "observation": torch.randn(batch_size, seq_len, 4).to(device),
        "cost_to_go": torch.ones(batch_size, seq_len, 1).to(device) * 0.1 # Very low budget
    }
    rl2s = torch.randn(batch_size, seq_len, 6).to(device)
    time_idxs = torch.arange(seq_len).repeat(batch_size, 1).unsqueeze(-1).to(device)
    
    with torch.no_grad():
        actions, _ = agent.get_actions(obs=obs, rl2s=rl2s, time_idxs=time_idxs, sample=True)
    
    print(f"Rollout actions (with low budget shielding): {actions.squeeze(-1)}")
    assert actions.shape == (batch_size, seq_len, 1)
    assert actions.dtype == torch.uint8

    # --- TEST 2: TRAINING PASS (FORWARD) ---
    print("\n--- Test 2: forward (Auxiliary Losses) ---")
    agent.train()
    # Batch expects L samples, where L-1 are transitions
    L = 5 
    mock_batch = Batch(
        obs={
            "observation": torch.randn(batch_size, L, 4).to(device),
            "cost_to_go": torch.randn(batch_size, L, 1).to(device)
        },
        rl2s=torch.randn(batch_size, L, 6).to(device),
        actions=torch.randn(batch_size, L-1, 5).to(device), # One-hot style for discrete
        rews=torch.randn(batch_size, L-1, 1).to(device),
        dones=torch.zeros(batch_size, L-1, 1).to(device),
        time_idxs=torch.arange(L).repeat(batch_size, 1).unsqueeze(-1).to(device),
        infos={
            "cost": torch.rand(batch_size, L-1).to(device),
            "soft_reset": torch.zeros(batch_size, L-1).to(device),
            "cost_limit": torch.ones(batch_size, 1).to(device)
        }
    )

    critic_loss, actor_loss = agent.forward(mock_batch, log_step=True)
    
    print("Auxiliary metrics logged:")
    for k in agent.update_info.keys():
        if k.startswith("shield/"):
            print(f"  {k}: {agent.update_info[k]:.4f}")
            
    # Check if all expected shield metrics are present
    required_keys = [
        "shield/wm_loss", "shield/mem_loss", "shield/latent_distill", 
        "shield/rnd_loss", "shield/eps_calib_loss", "shield/gamma_conjugacy_loss",
        "shield/aux_total_loss"
    ]
    for rk in required_keys:
        assert rk in agent.update_info, f"Missing metric: {rk}"

    # --- TEST 3: GRADIENT FLOW ---
    print("\n--- Test 3: Gradient Flow ---")
    # Total loss for optimization
    total_loss = critic_loss.mean() + actor_loss.mean()
    total_loss.backward()
    
    # Check gradients for auxiliary heads
    heads_to_check = {
        "proj_world": agent.proj_world.weight.grad,
        "dyn_z_w1": agent.dyn_z_w1.weight.grad,
        "mem_w1": agent.mem_w1.weight.grad,
        "rnd_pred_w1": agent.rnd_pred_w1.weight.grad,
        "eps_cal_scale_raw": agent.eps_cal_scale_raw.grad,
    }
    
    for name, grad in heads_to_check.items():
        if grad is not None:
            print(f"  {name} gradient norm: {grad.norm():.4e}")
            assert grad.norm() > 0, f"Zero gradient in {name}"
        else:
            print(f"  FAILED: {name} has no gradient!")
            assert False, f"{name} should have gradients"

    # Verify frozen params
    assert agent.rnd_target_w1.weight.grad is None, "Frozen RND target head should NOT have gradients"
    print("  Verified: RND target head is frozen (no gradient)")

    print("\nAll tests passed successfully!")

if __name__ == "__main__":
    test_shielded_agent_full()


def _make_mock_constrained_config():
    class MockConstrainedConfig:
        def __init__(self):
            self.fixed_lagrangian_multiplier = 1.0
            self.lambda_lr = 1e-3
            self.lambda_init = 0.0
            self.lambda_upper_bound = None
            self.max_unused_fraction = 1.0
            self.apply_cost_violation_mask = False

    return MockConstrainedConfig()


def _make_continuous_agent(**kwargs):
    obs_space = gym.spaces.Dict({
        "observation": gym.spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32),
        "cost_to_go": gym.spaces.Box(low=0, high=10, shape=(1,), dtype=np.float32),
    })
    rl2_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)
    action_space = gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
    return ShieldedAMAGOAgent(
        obs_space=obs_space,
        rl2_space=rl2_space,
        action_space=action_space,
        max_seq_len=8,
        tstep_encoder_type=FFTstepEncoder,
        traj_encoder_type=TformerTrajEncoder,
        constrained_config=_make_mock_constrained_config(),
        num_critics=2,
        **kwargs,
    )


def _make_eval_inputs():
    obs = {
        "observation": torch.zeros(1, 2, 3),
        "cost_to_go": torch.full((1, 2, 1), 0.1),
    }
    rl2s = torch.zeros(1, 2, 4)
    time_idxs = torch.arange(2).view(1, 2, 1)
    return obs, rl2s, time_idxs


def _attach_fake_eval_modules(agent, dist, critic_cost_fn):
    class FakeTstepEncoder(nn.Module):
        def forward(self, obs, rl2s, log_dict=None):
            return torch.zeros(obs["observation"].shape[0], obs["observation"].shape[1], 4)

    class FakeTrajEncoder(nn.Module):
        emb_dim = 4

        class _HiddenState:
            def __init__(self):
                self.key_cache = type("Cache", (), {"data": torch.zeros(1)})()
                self.val_cache = type("Cache", (), {"data": torch.zeros(1)})()

        def init_hidden_state(self, batch_size, device):
            return self._HiddenState()

        def forward(self, seq, time_idxs=None, hidden_state=None, log_dict=None):
            return seq, hidden_state

    class FakeActor(nn.Module):
        def forward(self, state_seq):
            return dist

    class FakeCostCritics(nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = []

        def forward(self, state_seq, action, log_dict=None):
            self.calls.append(tuple(action.shape))
            cost = critic_cost_fn(action)
            return cost.unsqueeze(3).repeat(1, 1, 1, 2, 1, 1)

    class IdentityPopArt(nn.Module):
        def forward(self, x, normalized=False):
            return x

    fake_critics = FakeCostCritics()
    agent.tstep_encoder = FakeTstepEncoder()
    agent.traj_encoder = FakeTrajEncoder()
    agent.actor = FakeActor()
    agent.cost_critics = fake_critics
    agent.cost_popart = IdentityPopArt()
    return fake_critics


def test_continuous_shield_does_not_fall_back_to_base_agent(monkeypatch):
    agent = _make_continuous_agent()
    agent.eval()
    agent.shield_continuous_num_samples = 4
    agent.shield_continuous_sampling_scheme = "iid"
    agent.shield_soft_beta = 10.0
    agent.cost_multiplier = 0.01

    def base_get_actions(*args, **kwargs):
        raise AssertionError("continuous shielding incorrectly fell back to Agent.get_actions")

    monkeypatch.setattr(Agent, "get_actions", base_get_actions)

    class FakeContinuousDist:
        def __init__(self):
            self.mean = torch.full((1, 2, len(agent.gammas), 2), 0.9)

        def rsample(self, sample_shape=torch.Size()):
            assert sample_shape == torch.Size()
            return torch.full((1, 2, len(agent.gammas), 2), -0.8)

        def sample(self, sample_shape=torch.Size()):
            if sample_shape:
                shape = tuple(sample_shape) + (1, 2, len(agent.gammas), 2)
                return torch.full(shape, -0.8)
            return self.rsample()

        def log_prob(self, action):
            return -((action - self.mean.unsqueeze(0)) ** 2)

    dist = FakeContinuousDist()
    _attach_fake_eval_modules(
        agent,
        dist=dist,
        critic_cost_fn=lambda action: torch.where(action[..., :1] > 0, 2.0, 0.0),
    )

    obs, rl2s, time_idxs = _make_eval_inputs()
    actions, _ = agent.get_actions(obs=obs, rl2s=rl2s, time_idxs=time_idxs, sample=False)

    assert actions.shape == (1, 2, 2)
    assert torch.all(actions < 0), actions


def test_continuous_shield_structured_candidates_cover_mean_local_and_wide():
    agent = _make_continuous_agent(shield_continuous_num_samples=4)
    agent.shield_continuous_sampling_scheme = "structured"
    agent.shield_continuous_wide_scale = 2.5

    class FakeContinuousDist:
        def __init__(self):
            self.mean = torch.tensor([[[[0.2, -0.1]]]], dtype=torch.float32)
            self.stddev = torch.tensor([[[[0.5, 0.25]]]], dtype=torch.float32)

    dist = FakeContinuousDist()
    torch.manual_seed(0)

    candidate_actions = agent._continuous_candidate_actions(dist)

    assert candidate_actions.shape == (4, 1, 1, 1, 2)
    assert torch.allclose(candidate_actions[0], dist.mean)
    assert torch.allclose(candidate_actions[1] + candidate_actions[2], 2.0 * dist.mean, atol=1e-6)
    wide_delta = (candidate_actions[3] - dist.mean).abs()
    local_delta = (candidate_actions[1] - dist.mean).abs()
    assert torch.any(wide_delta > local_delta)


def test_continuous_shield_structured_candidates_handle_unimplemented_stddev():
    agent = _make_continuous_agent(shield_continuous_num_samples=4)
    agent.shield_continuous_sampling_scheme = "structured"

    class FakeBaseDist:
        @property
        def stddev(self):
            raise NotImplementedError

        @property
        def scale(self):
            return torch.tensor([[[[0.3, 0.15]]]], dtype=torch.float32)

    class FakeContinuousDist:
        def __init__(self):
            self.mean = torch.tensor([[[[0.2, -0.1]]]], dtype=torch.float32)
            self.base_dist = FakeBaseDist()

        @property
        def stddev(self):
            raise NotImplementedError

    dist = FakeContinuousDist()
    torch.manual_seed(0)

    candidate_actions = agent._continuous_candidate_actions(dist)

    assert candidate_actions.shape == (4, 1, 1, 1, 2)


def test_continuous_final_candidate_sampling_temperature_scales_scores(monkeypatch):
    agent = _make_continuous_agent(shield_final_sampling_temperature=0.5)

    captured = {}

    def fake_multinomial(probs, num_samples):
        captured["probs"] = probs.clone()
        captured["num_samples"] = num_samples
        return torch.zeros((probs.shape[0], num_samples), dtype=torch.long, device=probs.device)

    monkeypatch.setattr(torch, "multinomial", fake_multinomial)

    candidate_actions = torch.tensor(
        [
            [[[[1.0, 0.0]]]],
            [[[[0.0, 1.0]]]],
        ],
        dtype=torch.float32,
    )
    scores = torch.tensor([[[2.0]], [[1.0]]], dtype=torch.float32)
    fallback_actions = torch.zeros((1, 1, 2), dtype=torch.float32)
    has_safe = torch.ones((1, 1), dtype=torch.bool)

    actions = agent._select_continuous_actions(
        candidate_actions=candidate_actions,
        scores=scores,
        fallback_actions=fallback_actions,
        has_safe=has_safe,
        sample=True,
    )

    expected_probs = torch.softmax(torch.tensor([[4.0, 2.0]], dtype=torch.float32), dim=-1)
    assert captured["num_samples"] == 1
    assert torch.allclose(captured["probs"], expected_probs)
    assert torch.allclose(actions, torch.tensor([[[1.0, 0.0]]], dtype=torch.float32))


def test_continuous_fast_path_early_exits_on_safe_mean_without_uncertainty_penalty():
    agent = _make_continuous_agent(
        shield_continuous_num_samples=5,
        shield_continuous_enable_fast_path=True,
        shield_continuous_mean_exit_margin=0.0,
        shield_continuous_stage1_num_samples=3,
        shield_uncertainty_mode="std",
        shield_uncertainty_kappa=10.0,
    )
    agent.eval()

    class FakeContinuousDist:
        def __init__(self):
            self.mean = torch.full((1, 2, len(agent.gammas), 2), 0.2)

        def sample(self, sample_shape=torch.Size()):
            if sample_shape:
                shape = tuple(sample_shape) + tuple(self.mean.shape)
                return torch.full(shape, 0.9)
            return self.mean.clone()

        def rsample(self, sample_shape=torch.Size()):
            return self.sample(sample_shape)

        def log_prob(self, action):
            return torch.zeros_like(action)

    dist = FakeContinuousDist()

    fake_critics = _attach_fake_eval_modules(
        agent,
        dist=dist,
        critic_cost_fn=lambda action: torch.full(action[..., :1].shape, 0.02),
    )
    agent._uncertainty_radius_candidates = lambda q_cost: torch.full(q_cost[..., 0, -1].shape, 0.2)

    obs, rl2s, time_idxs = _make_eval_inputs()
    actions, _ = agent.get_actions(obs=obs, rl2s=rl2s, time_idxs=time_idxs, sample=False)

    assert torch.allclose(actions, torch.full_like(actions, 0.2))
    assert fake_critics.calls == [(1, 1, 2, len(agent.gammas), 2)]


def test_continuous_fast_path_uses_stage1_without_full_expansion(monkeypatch):
    agent = _make_continuous_agent(
        shield_continuous_num_samples=5,
        shield_continuous_enable_fast_path=True,
        shield_continuous_mean_exit_margin=0.0,
        shield_continuous_stage1_num_samples=3,
    )
    agent.eval()
    agent.shield_continuous_sampling_scheme = "structured"
    agent.shield_filter_mode = "hard"
    agent.shield_filter_mode = "hard"

    class FakeContinuousDist:
        def __init__(self):
            self.mean = torch.tensor([[[[0.8, 0.8]] * len(agent.gammas)] * 2], dtype=torch.float32)
            self.stddev = torch.tensor([[[[0.4, 0.4]] * len(agent.gammas)] * 2], dtype=torch.float32)

        def sample(self, sample_shape=torch.Size()):
            if sample_shape:
                shape = tuple(sample_shape) + tuple(self.mean.shape)
                return torch.full(shape, -0.9)
            return torch.full_like(self.mean, -0.9)

        def rsample(self, sample_shape=torch.Size()):
            return self.sample(sample_shape)

        def log_prob(self, action):
            return -((action - self.mean.unsqueeze(0)) ** 2)

    dist = FakeContinuousDist()
    fake_critics = _attach_fake_eval_modules(
        agent,
        dist=dist,
        critic_cost_fn=lambda action: torch.where(action[..., :1] > 0.5, 2.0, 0.0),
    )

    stage1_candidates = torch.stack(
        [
            dist.mean,
            torch.full_like(dist.mean, 0.9),
            torch.full_like(dist.mean, -0.9),
        ],
        dim=0,
    )
    full_candidates = torch.cat(
        [stage1_candidates, torch.full((2, *dist.mean.shape), -0.9)],
        dim=0,
    )

    def fake_candidate_subset(action_dists, num_samples, base_action=None):
        if num_samples == 3:
            return stage1_candidates
        if num_samples == 5:
            return full_candidates
        raise AssertionError(f"unexpected num_samples={num_samples}")

    monkeypatch.setattr(agent, "_continuous_candidate_subset", fake_candidate_subset)

    obs, rl2s, time_idxs = _make_eval_inputs()
    actions, _ = agent.get_actions(obs=obs, rl2s=rl2s, time_idxs=time_idxs, sample=False)

    assert torch.all(actions < 0.8)
    assert fake_critics.calls == [
        (1, 1, 2, len(agent.gammas), 2),
        (3, 1, 2, len(agent.gammas), 2),
    ]


def test_continuous_fast_path_falls_back_to_full_expansion_when_stage1_has_no_safe_action(monkeypatch):
    agent = _make_continuous_agent(
        shield_continuous_num_samples=5,
        shield_continuous_enable_fast_path=True,
        shield_continuous_mean_exit_margin=0.0,
        shield_continuous_stage1_num_samples=3,
    )
    agent.eval()
    agent.shield_continuous_sampling_scheme = "structured"

    class FakeContinuousDist:
        def __init__(self):
            self.mean = torch.tensor([[[[0.8, 0.8]] * len(agent.gammas)] * 2], dtype=torch.float32)
            self.stddev = torch.tensor([[[[0.4, 0.4]] * len(agent.gammas)] * 2], dtype=torch.float32)

        def sample(self, sample_shape=torch.Size()):
            if sample_shape:
                shape = tuple(sample_shape) + tuple(self.mean.shape)
                return torch.full(shape, -0.9)
            return torch.full_like(self.mean, -0.9)

        def rsample(self, sample_shape=torch.Size()):
            return self.sample(sample_shape)

        def log_prob(self, action):
            return -((action - self.mean.unsqueeze(0)) ** 2)

    dist = FakeContinuousDist()

    def critic_cost_fn(action):
        unsafe = action[..., :1] > -0.5
        return torch.where(unsafe, 2.0, 0.0)

    fake_critics = _attach_fake_eval_modules(agent, dist=dist, critic_cost_fn=critic_cost_fn)

    stage1_candidates = torch.stack(
        [
            dist.mean,
            torch.full_like(dist.mean, 0.9),
            torch.full_like(dist.mean, 0.1),
        ],
        dim=0,
    )
    full_candidates = torch.cat(
        [stage1_candidates, torch.full((2, *dist.mean.shape), -0.9)],
        dim=0,
    )

    def fake_candidate_subset(action_dists, num_samples, base_action=None):
        if num_samples == 3:
            return stage1_candidates
        if num_samples == 5:
            return full_candidates
        raise AssertionError(f"unexpected num_samples={num_samples}")

    monkeypatch.setattr(agent, "_continuous_candidate_subset", fake_candidate_subset)

    obs, rl2s, time_idxs = _make_eval_inputs()
    actions, _ = agent.get_actions(obs=obs, rl2s=rl2s, time_idxs=time_idxs, sample=False)

    assert torch.all(actions <= 1.0)
    assert fake_critics.calls == [
        (1, 1, 2, len(agent.gammas), 2),
        (3, 1, 2, len(agent.gammas), 2),
        (5, 1, 2, len(agent.gammas), 2),
    ]


def test_continuous_fast_path_sample_true_early_exits_on_safe_sample_without_stage_expansion():
    agent = _make_continuous_agent(
        shield_continuous_num_samples=5,
        shield_continuous_enable_fast_path=True,
        shield_continuous_mean_exit_margin=0.0,
        shield_continuous_stage1_num_samples=3,
    )
    agent.eval()

    class FakeContinuousDist:
        def __init__(self):
            self.mean = torch.full((1, 2, len(agent.gammas), 2), 0.2)
            self.stddev = torch.full((1, 2, len(agent.gammas), 2), 0.1)
            self.sample_value = torch.full((1, 2, len(agent.gammas), 2), -0.4)

        def sample(self, sample_shape=torch.Size()):
            if sample_shape:
                shape = tuple(sample_shape) + tuple(self.mean.shape)
                return torch.full(shape, -0.7)
            return self.sample_value.clone()

        def rsample(self, sample_shape=torch.Size()):
            return self.sample(sample_shape)

        def log_prob(self, action):
            return torch.zeros_like(action)

    dist = FakeContinuousDist()
    fake_critics = _attach_fake_eval_modules(
        agent,
        dist=dist,
        critic_cost_fn=lambda action: torch.where(action[..., :1] < 0.0, 0.01, 2.0),
    )

    obs, rl2s, time_idxs = _make_eval_inputs()
    actions, _ = agent.get_actions(obs=obs, rl2s=rl2s, time_idxs=time_idxs, sample=True)

    assert torch.allclose(actions, torch.full_like(actions, -0.4))
    assert fake_critics.calls == [(1, 1, 2, len(agent.gammas), 2)]
