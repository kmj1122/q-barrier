import os
from typing import Optional, List
import numpy as np
import torch
import gymnasium as gym
import hydra
from omegaconf import DictConfig, OmegaConf
from amago.experiment import Experiment
from amago.agent import Agent as AMAGOAgent
from amago.shielded_agent import ShieldedAMAGOAgent
from amago import utils as amago_utils
from amago.nets.tstep_encoders import FFTstepEncoder
from amago.nets.traj_encoders import TformerTrajEncoder
from amago.loading import DiskTrajDataset
from amago.envs import AMAGOEnv
import envs
from utils.misc import set_seed
from utils.checkpoints import require_evaluation_checkpoint
from utils.generate_goals import train_test_goals_darkroom
import logging
import time
from functools import partial
from envs.dark_room.envs.darkroom import CostWrapper, DarkRoomGifWrapper
from amago.cli_utils import make_experiment_collect_only, make_experiment_learn_only
from amago.envs.amago_env import Trajectory
import itertools
from collections import deque
logging.captureWarnings(True)


class DarkroomObservationOnlyEncoder(FFTstepEncoder):
    """DarkRoom encoder variant that ignores cost-to-go in the learned latent.

    The shield still reads `obs["cost_to_go"]` directly at action selection time,
    but the timestep encoder only consumes the physical observation key.
    """

    def __init__(self, obs_space, rl2_space):
        super().__init__(obs_space=obs_space, rl2_space=rl2_space, specify_obs_keys=["observation"])


class CTGAblationWrapper(gym.ObservationWrapper):
    """Evaluation-only wrapper that intervenes on the observed cost-to-go signal."""

    def __init__(self, env: gym.Env, config):
        super().__init__(env)
        self.mode = self._normalize_mode(config.get("mode", "true"))
        self.constant_value = float(config.get("constant_value", 0.75))
        self.scale_factor = float(config.get("scale_factor", 1.0))
        self.shift_steps = int(config.get("shift_steps", 0))
        self.invert_max = float(config.get("invert_max", self.observation_space["cost_to_go"].high.item()))
        self.noise_std = float(config.get("noise_std", 0.0))
        self.clip_min = float(config.get("clip_min", self.observation_space["cost_to_go"].low.item()))
        self.clip_max = float(config.get("clip_max", self.observation_space["cost_to_go"].high.item()))
        self.rng = np.random.default_rng(int(config.get("seed", 0)))
        self.initial_ctg = None
        self.actual_history = deque()
        self.seen_initial_ctgs = []
        self.shuffled_episode_ctg = None
        if self.shift_steps < 0:
            raise ValueError("CTG shift mode currently supports non-negative lag only.")

    @staticmethod
    def _normalize_mode(raw_mode) -> str:
        if isinstance(raw_mode, bool):
            return "true" if raw_mode else "false"
        return str(raw_mode).strip().lower()

    def reset(self, **kwargs):
        self.actual_history.clear()
        self.initial_ctg = None
        self.shuffled_episode_ctg = None
        obs, info = self.env.reset(**kwargs)
        return self.observation(obs), info

    def observation(self, obs):
        obs = dict(obs)
        actual_ctg = np.float32(obs["cost_to_go"])
        if self.initial_ctg is None:
            self.initial_ctg = actual_ctg
            self.seen_initial_ctgs.append(float(actual_ctg))
            self.shuffled_episode_ctg = self._sample_episode_shuffle_value(float(actual_ctg))
        self.actual_history.append(float(actual_ctg))
        obs["cost_to_go"] = self._transform_cost_to_go(float(actual_ctg))
        return obs

    def _sample_episode_shuffle_value(self, fallback_value: float) -> np.float32:
        if self.mode != "episode_shuffle":
            return np.float32(fallback_value)
        if len(self.seen_initial_ctgs) <= 1:
            return np.float32(fallback_value)
        choices = self.seen_initial_ctgs[:-1]
        return np.float32(self.rng.choice(choices))

    def _transform_cost_to_go(self, actual_ctg: float) -> np.float32:
        if self.mode == "true":
            transformed = actual_ctg
        elif self.mode == "zero":
            transformed = 0.0
        elif self.mode == "constant":
            transformed = self.constant_value
        elif self.mode == "init_only":
            transformed = float(self.initial_ctg)
        elif self.mode == "scale":
            transformed = actual_ctg * self.scale_factor
        elif self.mode == "invert":
            transformed = self.invert_max - actual_ctg
        elif self.mode == "noise":
            transformed = actual_ctg + self.rng.normal(0.0, self.noise_std)
        elif self.mode == "shift":
            if len(self.actual_history) <= self.shift_steps:
                transformed = float(self.initial_ctg)
            else:
                transformed = self.actual_history[-(self.shift_steps + 1)]
        elif self.mode == "episode_shuffle":
            transformed = float(self.shuffled_episode_ctg)
        else:
            raise ValueError(f"Unsupported CTG ablation mode: {self.mode}")
        return np.float32(np.clip(transformed, self.clip_min, self.clip_max))


def test_ctg_ablation_modes_transform_cost_to_go():
    config = {
        "mode": "true",
        "constant_value": 0.75,
        "scale_factor": 0.5,
        "shift_steps": 1,
        "invert_max": 1.5,
    }
    wrapper = CTGAblationWrapper(_DummyCTGEnv(), config)

    obs, _ = wrapper.reset()
    assert obs["cost_to_go"] == np.float32(1.2)

    wrapper.mode = "zero"
    obs = wrapper.observation({"observation": np.array([0, 0], dtype=np.int32), "cost_to_go": np.float32(0.8)})
    assert obs["cost_to_go"] == np.float32(0.0)

    wrapper.mode = "constant"
    obs = wrapper.observation({"observation": np.array([0, 0], dtype=np.int32), "cost_to_go": np.float32(0.8)})
    assert obs["cost_to_go"] == np.float32(0.75)

    wrapper.mode = "init_only"
    obs = wrapper.observation({"observation": np.array([0, 0], dtype=np.int32), "cost_to_go": np.float32(0.4)})
    assert obs["cost_to_go"] == np.float32(1.2)

    wrapper.mode = "scale"
    obs = wrapper.observation({"observation": np.array([0, 0], dtype=np.int32), "cost_to_go": np.float32(0.8)})
    assert obs["cost_to_go"] == np.float32(0.4)

    wrapper.mode = "invert"
    obs = wrapper.observation({"observation": np.array([0, 0], dtype=np.int32), "cost_to_go": np.float32(0.3)})
    assert obs["cost_to_go"] == np.float32(1.2)


def test_ctg_ablation_shift_uses_history():
    wrapper = CTGAblationWrapper(
        _DummyCTGEnv(),
        {"mode": "shift", "shift_steps": 1, "invert_max": 1.5},
    )

    obs, _ = wrapper.reset()
    assert obs["cost_to_go"] == np.float32(1.2)

    obs, _, _, _, _ = wrapper.step(0)
    assert obs["cost_to_go"] == np.float32(1.2)

    obs, _, _, _, _ = wrapper.step(0)
    assert obs["cost_to_go"] == np.float32(1.0)


def test_ctg_ablation_accepts_boolean_true_mode():
    wrapper = CTGAblationWrapper(_DummyCTGEnv(), {"mode": True})

    obs, _ = wrapper.reset()

    assert obs["cost_to_go"] == np.float32(1.2)


class _DummyCTGEnv(gym.Env):
    def __init__(self):
        self.action_space = gym.spaces.Discrete(1)
        self.observation_space = gym.spaces.Dict(
            {
                "observation": gym.spaces.Box(low=0, high=10, shape=(2,), dtype=np.int32),
                "cost_to_go": gym.spaces.Box(low=0.0, high=1.5, shape=(), dtype=np.float32),
            }
        )
        self._obs = [
            np.float32(1.2),
            np.float32(1.0),
            np.float32(0.7),
        ]
        self._idx = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._idx = 0
        return self._make_obs(self._obs[self._idx]), {}

    def step(self, action):
        self._idx = min(self._idx + 1, len(self._obs) - 1)
        return self._make_obs(self._obs[self._idx]), 0.0, False, False, {}

    def _make_obs(self, cost_to_go):
        return {
            "observation": np.array([0, 0], dtype=np.int32),
            "cost_to_go": np.float32(cost_to_go),
        }

class InnerEpisodeTrendsMetric:
    def __init__(self, config: DictConfig):
        self.config = config
        self.trajectories = []
    
    def hook(self, env: AMAGOEnv, trajectory: Trajectory, episode_finished: bool):
        if episode_finished:
            self.trajectories.append(trajectory)

    def save_metrics(self):
        soft_resets = np.array(list(itertools.zip_longest(*[[ts.info["soft_reset"].item() for ts in t] for t in self.trajectories], fillvalue=0))).T
        rewards = np.array(list(itertools.zip_longest(*[[ts.reward.item() for ts in t] for t in self.trajectories], fillvalue=0))).T
        costs = np.array(list(itertools.zip_longest(*[[ts.info["cost"].item() for ts in t] for t in self.trajectories], fillvalue=0))).T
        cost_limits = np.array(list(itertools.zip_longest(*[[ts.info["cost_limit"].item() for ts in t] for t in self.trajectories], fillvalue=0))).T
        rewards_acc = np.cumsum(rewards, axis=1)
        costs_acc = np.cumsum(costs, axis=1)
        take_mask = soft_resets==1
        raw_rets = rewards_acc[take_mask].reshape(take_mask.shape[0], take_mask[0].sum())
        raw_costs = costs_acc[take_mask].reshape(take_mask.shape[0], take_mask[0].sum())
        cost_limits = cost_limits[take_mask].reshape(take_mask.shape[0], take_mask[0].sum())
        rets = np.diff(raw_rets, axis=1, prepend=0)
        costs = np.diff(raw_costs, axis=1, prepend=0)
        np.savez(os.path.join(self.config.ckpt_base_dir, self.config.run_name, self.config.metrics_file_name), rets=rets, costs=costs, cost_limits=cost_limits)
        

def post_process_config(config: DictConfig) -> DictConfig:
    if not hasattr(config, "eval_ctg_ablation") or config.eval_ctg_ablation is None:
        config.eval_ctg_ablation = OmegaConf.create({"mode": "true"})
    config['parallel_actors'] = config.num_test_goals
    assert isinstance(config.parallel_actors, int)
    return config


def load_test_checkpoint(experiment: Experiment, checkpoint_path: str, allow_legacy_shield_aux: bool) -> None:
    if not allow_legacy_shield_aux:
        experiment.load_checkpoint_from_path(checkpoint_path, is_accelerate_state=False)
        return

    ckpt = amago_utils.retry_load_checkpoint(checkpoint_path, map_location=experiment.DEVICE)
    incompatible = experiment.policy.load_state_dict(ckpt, strict=False)
    allowed_missing = {
        "mem_cost_head.weight",
        "mem_cost_head.bias",
        "mem_cost_out.weight",
        "mem_cost_out.bias",
    }
    bad_missing = sorted(set(incompatible.missing_keys) - allowed_missing)
    unexpected = sorted(incompatible.unexpected_keys)
    if bad_missing or unexpected:
        raise RuntimeError(
            "Checkpoint is incompatible with the current shielded agent. "
            f"Missing keys: {bad_missing or incompatible.missing_keys}. "
            f"Unexpected keys: {unexpected}."
        )
    if incompatible.missing_keys:
        print(
            "Loading legacy shield checkpoint with default-initialized eval-only aux heads: "
            f"{sorted(incompatible.missing_keys)}"
        )

@hydra.main(version_base=None, config_path=".", config_name="test_crl2_darkroom_config")
def main(config: DictConfig) -> None:
    """
    Test script for evaluating trained models.
    Override the run_name with the time stamped run_name in the config file of the original experiment.
    """
    OmegaConf.set_struct(config, False)
    config = OmegaConf.merge(config, OmegaConf.from_cli())
    config = post_process_config(config)
    OmegaConf.set_struct(config, True)

    config.load_checkpoint_path = require_evaluation_checkpoint(config)
    
    # Create experiment for testing/evaluation purposes
    set_seed(config.seed)
    
    train_goals, test_goals = train_test_goals_darkroom(
        grid_size=config.grid_size, 
        num_train_goals=config.num_train_goals, 
        num_test_goals=config.num_test_goals, 
        seed=config.seed,
        empty_radius=config.constrained_env_config.empty_radius,
        train_radial_alpha=config.constrained_env_config.train_radial_alpha,
    )
    
    os.makedirs(config.ckpt_base_dir, exist_ok=True)
    dataset = DiskTrajDataset(
        dset_root=config.ckpt_base_dir,
        dset_name=config.run_name,
        dset_max_size=config.replay_capacity,
    )

    def make_env(goal, index, get_experiment, save_gif=False):
        env = gym.make(
            config.env_name,
            goal_pos=goal,
            size=config.grid_size,
            max_episodes=config.max_inner_episodes,
            time_limit=config.time_limit,
        )
        if config.constrained_env_config.cost_seed == 'dynamic':
            seed = np.random.randint(0, 1000000)
        else:
            seed = config.constrained_env_config.cost_seed
        # Always use negative radial_alpha for validation (test) environments
        radial_alpha = -config.constrained_env_config.train_radial_alpha
        
        # Override reset_to_new_cost if fix_map_across_budgets is true
        reset_to_new_cost = config.constrained_env_config.reset_to_new_cost
        if getattr(config.constrained_env_config, "fix_map_across_budgets", False):
            reset_to_new_cost = False
        
        env = CostWrapper(
            env=env,
            num=config.constrained_env_config.cost_num,
            cost=config.constrained_env_config.cost_cost,
            seed=seed + index,
            min_cost_to_go=config.constrained_env_config.min_cost_to_go,
            max_cost_to_go=config.constrained_env_config.max_cost_to_go,
            guranatee_path=config.constrained_env_config.guranatee_path,
            reset_to_new_cost=reset_to_new_cost,
            radial_alpha=radial_alpha,
            exclude_goal_and_start_from_cost=getattr(config.constrained_env_config, 'exclude_goal_and_start_from_cost', False),
        )
        eval_ctg_ablation = OmegaConf.to_container(config.eval_ctg_ablation, resolve=True)
        if str(eval_ctg_ablation.get("mode", "true")) != "true":
            eval_ctg_ablation.setdefault("seed", int(config.seed) + index)
            env = CTGAblationWrapper(env=env, config=eval_ctg_ablation)
        env = DarkRoomGifWrapper(
            env=env,
            env_name=config.env_name,
            get_experiment=get_experiment,
            enabled=save_gif and index == 0,  # Only enable GIF creation for first env
            save_dir=config.ckpt_base_dir,
            run_name=config.run_name,
            add_process_handle=add_process_handle,
            name_postfix=config.name_postfix,
        )
        return env

    process_handles = []
    def add_process_handle(process):
        process_handles.append(process)

    def get_experiment():
        return experiment  

    make_train_env = [partial(make_env, goal, index, get_experiment, save_gif=config.save_gifs) for index, goal in enumerate(train_goals)]
    make_train_env = make_train_env * config.parallel_actors_per_goal
    make_val_env = [partial(make_env, goal, index, get_experiment, save_gif=config.save_gifs) for index, goal in enumerate(test_goals)]

    inner_episode_trends_metric = InnerEpisodeTrendsMetric(config)

    policy_kwargs = {}
    if getattr(config, "use_shield", False):
        agent_cls = ShieldedAMAGOAgent
        if hasattr(config, "shield_config"):
            policy_kwargs = OmegaConf.to_container(config.shield_config, resolve=True)
    else:
        agent_cls = AMAGOAgent

    tstep_encoder_cls = (
        DarkroomObservationOnlyEncoder
        if bool(getattr(config, "exclude_cost_to_go_from_encoder", False))
        else FFTstepEncoder
    )
    
    experiment = Experiment(
        run_name=config.run_name,
        ckpt_base_dir=config.ckpt_base_dir,
        max_seq_len=config.max_seq_len,
        dataset=dataset,
        tstep_encoder_type=tstep_encoder_cls,
        traj_encoder_type=TformerTrajEncoder,
        agent_type=agent_cls,
        make_train_env=make_train_env,
        make_val_env=make_val_env,
        val_timesteps_per_epoch=config.val_timesteps_per_epoch,
        parallel_actors=config.parallel_actors,
        exploration_wrapper_type=None,
        epochs=config.epochs,
        start_learning_at_epoch=config.start_learning_at_epoch,
        train_timesteps_per_epoch=config.train_timesteps_per_epoch,
        train_batches_per_epoch=config.train_batches_per_epoch,
        val_interval=config.val_interval,
        ckpt_interval=config.ckpt_interval,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        log_to_wandb=False,  # Disable wandb for testing
        wandb_project=config.wandb_project,
        mixed_precision=config.mixed_precision,
        env_mode=config.env_mode,
        constrained_config=config.constrained_agent_config,
        on_trajectory_finish=inner_episode_trends_metric.hook,
        policy_kwargs=policy_kwargs,
    )
    
    experiment.start()
    if getattr(config, "load_checkpoint_path", None):
        checkpoint_path = config.load_checkpoint_path
    else:
        checkpoint_path = os.path.join(
            experiment.ckpt_dir, "policy_weights", f"policy_epoch_{config.load_checkpoint_epoch}.pt"
        )
    load_test_checkpoint(
        experiment,
        checkpoint_path,
        allow_legacy_shield_aux=getattr(config, "use_shield", False),
    )
    experiment.evaluate_val()
    print("Evaluation completed successfully!")
    for process in process_handles:
        process.join()
    
    if config.save_metrics:
        inner_episode_trends_metric.save_metrics()

if __name__ == "__main__":
    main()
