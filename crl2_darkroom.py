import os
import re
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
from envs.dark_room.envs.darkroom import CostWrapper, DarkRoomGifWrapper
from amago.cli_utils import make_experiment_collect_only, make_experiment_learn_only
logging.captureWarnings(True)


class DarkroomObservationOnlyEncoder(FFTstepEncoder):
    """DarkRoom encoder variant that ignores cost-to-go in the learned latent.

    The shield still reads `obs["cost_to_go"]` directly at action selection time,
    but the timestep encoder only consumes the physical observation key.
    """

    def __init__(self, obs_space, rl2_space):
        super().__init__(obs_space=obs_space, rl2_space=rl2_space, specify_obs_keys=["observation"])

def post_process_config(config: DictConfig) -> DictConfig:
    deterministic = bool(getattr(config, "deterministic", False))
    if deterministic and config.env_mode == "async":
        config.env_mode = "sync"
    if (
        deterministic
        and config.constrained_env_config.cost_seed == "dynamic"
    ):
        config.constrained_env_config.cost_seed = int(config.seed)
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


def infer_epoch_from_checkpoint_path(path: str) -> Optional[int]:
    match = re.search(r"epoch_(\d+)", os.path.basename(path))
    if match is None:
        return None
    return int(match.group(1))

@hydra.main(version_base=None, config_path=".", config_name="crl2_darkroom_config")
def main(config: DictConfig) -> None:
    OmegaConf.set_struct(config, False)
    config = OmegaConf.merge(config, OmegaConf.from_cli())
    config = post_process_config(config) 
    OmegaConf.set_struct(config, True)
    
    set_seed(config.seed, deterministic=bool(getattr(config, "deterministic", False)))
    
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

    def make_env(goal, index, get_experiment, is_validation=False):
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
        # Use negative radial_alpha for validation environments
        radial_alpha = -config.constrained_env_config.train_radial_alpha if is_validation else config.constrained_env_config.train_radial_alpha
        
        env = CostWrapper(
            env=env,
            num=config.constrained_env_config.cost_num,
            cost=config.constrained_env_config.cost_cost,
            seed=seed + index,
            min_cost_to_go=config.constrained_env_config.min_cost_to_go,
            max_cost_to_go=config.constrained_env_config.max_cost_to_go,
            guranatee_path=config.constrained_env_config.guranatee_path,
            reset_to_new_cost=config.constrained_env_config.reset_to_new_cost,
            radial_alpha=radial_alpha,
            exclude_goal_and_start_from_cost=getattr(config.constrained_env_config, 'exclude_goal_and_start_from_cost', False),
        )
        env = DarkRoomGifWrapper(
            env=env,
            env_name=config.env_name,
            get_experiment=get_experiment,
            enabled=is_validation and index == 0,  # Only enable GIF creation for first validation env
            save_dir=config.ckpt_base_dir,
            run_name=config.run_name,
        )
        return env

    # Create provider function that captures local scope for experiment
    def get_experiment():
        return experiment  # Will be defined later in this scope

    # Create environment creation functions
    make_train_env = [partial(make_env, goal, index, get_experiment, is_validation=False) for index, goal in enumerate(train_goals)]
    make_val_env = [partial(make_env, goal, index, get_experiment, is_validation=True) for index, goal in enumerate(test_goals)]
    
    # Create the experiment
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
        policy_kwargs=policy_kwargs,
        **({
            "dloader_workers": config.dloader_workers,
        } if hasattr(config, "dloader_workers") else {}),
    )

    experiment = configure_training_mode(experiment, config)
   
    experiment.start()
    if hasattr(config, "resume_from_epoch") and config.resume_from_epoch is not None:
        experiment.load_checkpoint(
            epoch=int(config.resume_from_epoch),
            resume_training_state=bool(getattr(config, "resume_training_state", True)),
        )
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
