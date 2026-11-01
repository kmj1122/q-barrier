import os
import re
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
from amago.nets.tstep_encoders import FFTstepEncoder
from amago.nets.traj_encoders import TformerTrajEncoder
from amago.loading import DiskTrajDataset
from amago.envs import AMAGOEnv
import envs
from utils.misc import set_seed
from utils.generate_goals import train_test_goals_darkroom
import logging
from functools import partial
from envs.mujoco.mujoco import MujocoWrapper, MujocoVideoRecorder
from amago.cli_utils import make_experiment_collect_only, make_experiment_learn_only
logging.captureWarnings(True)


class MujocoObservationOnlyEncoder(FFTstepEncoder):
    """Mujoco encoder variant that excludes cost-to-go from the learned latent."""

    def __init__(self, obs_space, rl2_space):
        super().__init__(obs_space=obs_space, rl2_space=rl2_space, specify_obs_keys=["observation"])

def post_process_config(config: DictConfig) -> DictConfig:
    deterministic = bool(getattr(config, "deterministic", False))
    if deterministic and config.env_mode == "async":
        config.env_mode = "sync"
    if deterministic:
        config.dloader_workers = 0
    return config


def configure_training_mode(experiment, config):
    """Apply enabled mode overrides, retaining collect-only precedence."""
    if bool(getattr(config, "collect_only", False)):
        return make_experiment_collect_only(experiment)
    if bool(getattr(config, "learn_only", False)):
        return make_experiment_learn_only(experiment)
    return experiment


def resume_training(experiment, config):
    """Resume from the same checkpoint layout used by Experiment's save helpers."""
    epoch = int(config.resume_from_epoch)
    resume_state = bool(getattr(config, "resume_training_state", True))
    load_run_name = config.get("load_run_name", config.run_name)
    load_base = config.get("load_ckpt_base_dir", config.ckpt_base_dir)
    if load_run_name == config.run_name and load_base == config.ckpt_base_dir:
        experiment.load_checkpoint(epoch=epoch, resume_training_state=resume_state)
        return
    checkpoint_dir = os.path.join(load_base, load_run_name, "ckpts")
    if resume_state:
        source_path = os.path.join(
            checkpoint_dir, "training_states", f"{load_run_name}_epoch_{epoch}"
        )
    else:
        source_path = os.path.join(
            checkpoint_dir, "policy_weights", f"policy_epoch_{epoch}.pt"
        )
    experiment.load_checkpoint_from_path(source_path, is_accelerate_state=resume_state)
    experiment.epoch = epoch


def infer_epoch_from_checkpoint_path(path: str) -> Optional[int]:
    match = re.search(r"epoch_(\d+)", os.path.basename(path))
    if match is None:
        return None
    return int(match.group(1))

@hydra.main(version_base=None, config_path=".", config_name="crl2_mujoco_config")
def main(config: DictConfig) -> None:
    OmegaConf.set_struct(config, False)
    config = OmegaConf.merge(config, OmegaConf.from_cli())
    config = post_process_config(config) 
    OmegaConf.set_struct(config, True)
    
    set_seed(config.seed, deterministic=bool(getattr(config, "deterministic", False)))
    
    os.makedirs(config.ckpt_base_dir, exist_ok=True)
    dataset = DiskTrajDataset(
        dset_root=config.ckpt_base_dir,
        dset_name=config.run_name,
        dset_max_size=config.replay_capacity,
    )

    shared_epoch = multiprocessing.Value('i', 0)

    def make_env(index, is_validation=False):
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
                render_mode="rgb_array", camera_name="fixedfar", width=512, height=512
            )
            env = MujocoVideoRecorder(
                env=env,
                output_dir=config.ckpt_base_dir,
                enabled=True,
                epoch_shared=shared_epoch,
                run_name=config.run_name,
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

    # Create environment creation functions
    make_train_env = [partial(make_env, index, is_validation=False) for index in range(config.num_train_goals)]
    make_val_env = [partial(make_env, index, is_validation=True) for index in range(config.num_test_goals)]
    
    # Create the experiment
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

    tstep_encoder_cls = (
        MujocoObservationOnlyEncoder
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
        exploration_wrapper_type=None,
        epochs=config.epochs,
        start_learning_at_epoch=config.start_learning_at_epoch,
        train_timesteps_per_epoch=config.train_timesteps_per_epoch,
        train_batches_per_epoch=config.train_batches_per_epoch,
        val_interval=config.val_interval,
        ckpt_interval=config.ckpt_interval,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        log_to_wandb=config.use_wandb,
        wandb_project=config.wandb_project,
        mixed_precision=config.mixed_precision,
        env_mode=config.env_mode,
        constrained_config=config.constrained_agent_config,
        # Provide shared epoch to experiment so it updates it each epoch
        shared_epoch=shared_epoch,
        has_dset_edit_rights=hasattr(config, 'has_dset_edit_rights') and config.has_dset_edit_rights,
        policy_kwargs=policy_kwargs,
        **({
            "dloader_workers": config.dloader_workers,
        } if hasattr(config, "dloader_workers") else {}),
        **({
            "lr_warmup_steps": config.lr_warmup_steps,
        } if hasattr(config, 'lr_warmup_steps') else {}),
    )

    experiment = configure_training_mode(experiment, config)
   
    experiment.start()
    
    if hasattr(config, "resume_from_epoch") and config.resume_from_epoch is not None:
        resume_training(experiment, config)
    elif hasattr(config, "load_checkpoint_path") and config.load_checkpoint_path is not None:
        is_accelerate_state = bool(getattr(config, "load_checkpoint_is_accelerate_state", False))
        experiment.load_checkpoint_from_path(
            config.load_checkpoint_path,
            is_accelerate_state=is_accelerate_state,
        )
        loaded_epoch = getattr(config, "load_checkpoint_epoch", None)
        if loaded_epoch is None:
            loaded_epoch = infer_epoch_from_checkpoint_path(config.load_checkpoint_path)
        if loaded_epoch is not None:
            experiment.epoch = int(loaded_epoch)
    
    if config.use_wandb:
        import wandb
        hydra_config_dict = OmegaConf.to_container(config, resolve=True)
        wandb.config.update({"hydra_config": hydra_config_dict})

    experiment.learn()
    
if __name__ == "__main__":
    main()
