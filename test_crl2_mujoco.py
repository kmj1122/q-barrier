import os
import multiprocessing
from typing import Optional, List
import numpy as np
import torch
import gymnasium as gym
import hydra
from omegaconf import DictConfig, OmegaConf
OmegaConf.register_new_resolver("eval", eval)
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
import logging
from functools import partial
from envs.mujoco.mujoco import MujocoWrapper, MujocoVideoRecorder
from amago.envs.amago_env import Trajectory
import itertools

logging.captureWarnings(True)


class InnerEpisodeTrendsMetric:
    def __init__(self, config: DictConfig):
        self.config = config
        self.trajectories = []
    
    def hook(self, env: AMAGOEnv, trajectory: Trajectory, episode_finished: bool):
        if episode_finished:
            self.trajectories.append(trajectory)

    def save_metrics(self):
        soft_resets = np.array(list(itertools.zip_longest(*[[ts.info["soft_reset"].item() for ts in t] for t in self.trajectories], fillvalue=0))).T
        # Use unnormalized original_reward if available, otherwise fall back to ts.reward
        rewards = np.array(list(itertools.zip_longest(*[[(ts.info["original_reward"].item() if "original_reward" in ts.info else ts.reward.item()) for ts in t] for t in self.trajectories], fillvalue=0))).T
        costs = np.array(list(itertools.zip_longest(*[[ts.info["cost"].item() for ts in t] for t in self.trajectories], fillvalue=0))).T
        cost_limits = np.array(list(itertools.zip_longest(*[[ts.info["cost_limit"].item() for ts in t] for t in self.trajectories], fillvalue=0))).T

        # Extract velocity thresholds if present (for velocity tasks)
        velocity_thresholds = None
        if len(self.trajectories) > 0 and "velocity_threshold" in self.trajectories[0][0].info:
            velocity_thresholds = np.array(list(itertools.zip_longest(*[[ts.info["velocity_threshold"].item() for ts in t] for t in self.trajectories], fillvalue=0))).T

        rewards_acc = np.cumsum(rewards, axis=1)
        costs_acc = np.cumsum(costs, axis=1)
        take_mask = soft_resets==1
        raw_rets = rewards_acc[take_mask].reshape(take_mask.shape[0], take_mask[0].sum())
        raw_costs = costs_acc[take_mask].reshape(take_mask.shape[0], take_mask[0].sum())
        cost_limits = cost_limits[take_mask].reshape(take_mask.shape[0], take_mask[0].sum())
        rets = np.diff(raw_rets, axis=1, prepend=0)
        costs = np.diff(raw_costs, axis=1, prepend=0)

        # Save metrics, including velocity thresholds if present
        save_dict = {'rets': rets, 'costs': costs, 'cost_limits': cost_limits}
        if velocity_thresholds is not None:
            velocity_thresholds_reshaped = velocity_thresholds[take_mask].reshape(take_mask.shape[0], take_mask[0].sum())
            save_dict['velocity_thresholds'] = velocity_thresholds_reshaped

        np.savez(os.path.join(self.config.ckpt_base_dir, self.config.run_name, self.config.metrics_file_name), **save_dict)


def post_process_config(config: DictConfig) -> DictConfig:
    if not hasattr(config, "sample_actions"):
        config.sample_actions = False
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
    allowed_unexpected_prefixes = (
        "eps_cal_",
        "proj_policy.",
        "proj_world.",
        "proj_cbf.",
        "dyn_z_w1.",
        "dyn_z_w2.",
        "dyn_r.",
        "dyn_c1.",
        "dyn_c2.",
        "dyn_logvar.",
        "mem_w1.",
        "mem_w2.",
        "rnd_target_w1.",
        "rnd_target_w2.",
        "rnd_pred_w1.",
        "rnd_pred_w2.",
    )
    bad_missing = sorted(set(incompatible.missing_keys) - allowed_missing)
    unexpected = sorted(
        key
        for key in incompatible.unexpected_keys
        if not key.startswith(allowed_unexpected_prefixes)
    )
    if bad_missing or unexpected:
        raise RuntimeError(
            "Checkpoint is incompatible with the current eval agent. "
            f"Missing keys: {bad_missing or incompatible.missing_keys}. "
            f"Unexpected keys: {unexpected}."
        )
    if incompatible.missing_keys:
        print(
            "Loading legacy shield checkpoint with default-initialized eval-only aux heads: "
            f"{sorted(incompatible.missing_keys)}"
        )


@hydra.main(version_base=None, config_path=".", config_name="test_crl2_mujoco_config")
def main(config: DictConfig) -> None:
    """
    Test script for evaluating trained Mujoco models.
    Loads a checkpoint and records a GIF/video for a single validation episode.
    Ensure your config sets max_inner_episodes=1 and an appropriate time_limit.
    """
    OmegaConf.set_struct(config, False)
    config = OmegaConf.merge(config, OmegaConf.from_cli())
    config = post_process_config(config)
    OmegaConf.set_struct(config, True)

    config.load_checkpoint_path = require_evaluation_checkpoint(config)

    set_seed(config.seed)

    dataset = DiskTrajDataset(
        dset_root=config.ckpt_base_dir,
        dset_name=config.run_name,
        dset_max_size=config.replay_capacity,
    )

    shared_epoch = multiprocessing.Value('i', 0)

    def make_env(index: int, is_validation: bool = False):
        # Get task type from config (defaults to "goal" for backward compatibility)
        task_type = config.get('task_type', 'goal')

        # Task-specific parameters
        if task_type == 'goal':
            radial_alpha = -config.constrained_env_config.train_radial_alpha if is_validation else config.constrained_env_config.train_radial_alpha
        else:  # velocity task
            radial_alpha = None

        if is_validation and index == 0 and not hasattr(config, 'no_video'):
            env = MujocoWrapper(
                config.env_name,
                seed=config.seed + index,
                time_limit=config.time_limit,
                max_episodes=config.max_inner_episodes,
                min_cost_to_go=config.constrained_env_config.min_ctg,
                max_cost_to_go=config.constrained_env_config.max_ctg,
                radial_alpha=radial_alpha,
                task_type=task_type,
                normalize_reward=config.normalize_reward,
                sticky_movement_multiplier=config.sticky_movement_multiplier,
                lidar=config.lidar,
                render_mode="rgb_array",
                camera_name="fixedfar",
                width=512,
                height=512,
            )
            env = MujocoVideoRecorder(
                env=env,
                output_dir=config.ckpt_base_dir,
                enabled=True,
                epoch_shared=shared_epoch,
                run_name=config.run_name,
                name_postfix=config.name_postfix,
            )
        else:
            env = MujocoWrapper(
                config.env_name,
                seed=config.seed + index,
                time_limit=config.time_limit,
                max_episodes=config.max_inner_episodes,
                min_cost_to_go=config.constrained_env_config.min_ctg,
                max_cost_to_go=config.constrained_env_config.max_ctg,
                radial_alpha=radial_alpha,
                task_type=task_type,
                normalize_reward=config.normalize_reward,
                sticky_movement_multiplier=config.sticky_movement_multiplier,
                lidar=config.lidar,
            )
        env = AMAGOEnv(env, env_name=config.env_name)
        return env

    make_train_env = [partial(make_env, 0, is_validation=False)]
    make_val_env = [partial(make_env, index, is_validation=True) for index in range(config.num_test_goals)]

    inner_episode_trends_metric = InnerEpisodeTrendsMetric(config)

    policy_kwargs = {
        "reward_multiplier": config.reward_multiplier,
        "cost_multiplier": config.cost_multiplier,
    }
    if getattr(config, "use_shield", False):
        agent_cls = ShieldedAMAGOAgent
        if hasattr(config, "shield_config"):
            policy_kwargs.update(OmegaConf.to_container(config.shield_config, resolve=True))
    else:
        agent_cls = AMAGOAgent

    experiment = Experiment(
        run_name=config.run_name,
        ckpt_base_dir=config.ckpt_base_dir,
        max_seq_len=config.max_seq_len,
        dataset=dataset,
        tstep_encoder_type=FFTstepEncoder,
        traj_encoder_type=TformerTrajEncoder,
        agent_type=agent_cls,
        make_train_env=make_train_env,
        make_val_env=make_val_env,
        val_timesteps_per_epoch=config.val_timesteps_per_epoch,
        exploration_wrapper_type=None,
        epochs=config.epochs,
        start_learning_at_epoch=config.start_learning_at_epoch,
        train_timesteps_per_epoch=config.train_timesteps_per_epoch,
        train_batches_per_epoch=config.train_batches_per_epoch,
        val_interval=config.val_interval,
        ckpt_interval=config.ckpt_interval,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        log_to_wandb=False,
        wandb_project=config.wandb_project,
        mixed_precision=config.mixed_precision,
        env_mode=config.env_mode,
        constrained_config=config.constrained_agent_config,
        on_trajectory_finish=inner_episode_trends_metric.hook,
        shared_epoch=shared_epoch,
        sample_actions=bool(config.sample_actions),
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
        allow_legacy_shield_aux=getattr(config, "allow_legacy_shield_aux", False)
        or getattr(config, "use_shield", False),
    )

    # Run validation; to record a single episode, configure max_inner_episodes=1 and set
    # val_timesteps_per_epoch to be >= environment time_limit.
    experiment.evaluate_val()
    print("Evaluation (with recording) completed successfully!")

    if config.save_metrics:
        inner_episode_trends_metric.save_metrics()

    # Dump executed-action barrier margins and per-step candidate feasibility.
    try:
        _policy = experiment.policy
        if getattr(_policy, "shield_log_margins", False) and hasattr(_policy, "dump_margin_log"):
            _out = os.path.join(config.ckpt_base_dir, config.run_name, "executed_margins.npz")
            os.makedirs(os.path.dirname(_out), exist_ok=True)
            _n = _policy.dump_margin_log(_out)
            print(f"[margins] wrote {_n} executed-action margin rows to {_out}")
    except Exception as _e:
        print(f"[margins] skipped: {_e}")


if __name__ == "__main__":
    main()
