"""CPU checkpoint compatibility and one reset-observation inference; never step environments."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
NAMES = ("darkroom", "point", "car", "ant", "cheetah")


def check_entrypoint(name):
    """Exercise production startup/loading, replacing evaluation before any interaction."""
    import importlib
    import torch
    from omegaconf import OmegaConf
    from amago.experiment import Experiment
    module_name = "test_crl2_darkroom" if name == "darkroom" else "test_crl2_mujoco"
    module = importlib.import_module(module_name)
    cfg = OmegaConf.load(ROOT / "pretrained" / f"{name}.yaml")
    checkpoint = (ROOT / "pretrained" / f"{name}.pt").resolve()
    if name not in NAMES or not checkpoint.is_relative_to(ROOT):
        raise ValueError("Unrecognized checkpoint")
    manifest = json.loads((ROOT / "pretrained/manifest.json").read_text())
    entries = [item for item in manifest["files"] if item["path"] == f"pretrained/{name}.pt"]
    if manifest.get("format") != 1 or len(entries) != 1:
        raise ValueError("Missing or ambiguous manifest entry")
    if checkpoint.stat().st_size != entries[0]["bytes"] or hashlib.sha256(checkpoint.read_bytes()).hexdigest() != entries[0]["sha256"]:
        raise ValueError("Checkpoint integrity check failed")
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    instances = []
    calls = []
    original_start = Experiment.start
    original_evaluate = Experiment.evaluate_val

    def tracked_start(self, *args, **kwargs):
        instances.append(self)
        return original_start(self, *args, **kwargs)

    def verify_only(self, *args, **kwargs):
        actual = self.policy.state_dict()
        if actual.keys() != state.keys():
            raise ValueError("Production loader did not preserve exactly the checkpoint keys")
        unequal = [key for key in state if not torch.equal(actual[key].detach().cpu(), state[key])]
        if unequal:
            raise ValueError(f"Production policy differs from saved tensors: {unequal}")
        calls.append(True)
        return None

    try:
        Experiment.start = tracked_start
        Experiment.evaluate_val = verify_only
        with tempfile.TemporaryDirectory(prefix="qbarrier-startup-") as temporary:
            cfg.ckpt_base_dir = temporary
            cfg.run_name = "startup_check"
            cfg.num_train_goals = 1
            cfg.num_test_goals = 1
            cfg.env_mode = "sync"
            cfg.dloader_workers = 0
            cfg.save_gifs = False
            cfg.no_video = True
            cfg.save_metrics = False
            cfg.load_checkpoint_path = str(checkpoint)
            sys.argv = [str(ROOT / f"{module_name}.py")]
            module.main.__wrapped__(cfg)
            if len(calls) != 1:
                raise ValueError(f"Expected one replaced evaluation call, got {len(calls)}")
    finally:
        for instance in instances:
            for attr in ("train_envs", "val_envs"):
                environment = getattr(instance, attr, None)
                if environment is not None:
                    environment.close()
        Experiment.start = original_start
        Experiment.evaluate_val = original_evaluate
    return {"entrypoint": module_name, "production_state_exact": True,
            "replaced_evaluation_calls": len(calls), "environment_steps": 0,
            "scope": "production startup and loading only; evaluation replaced by tensor comparison"}


def check(name):
    import numpy as np
    import torch
    import gymnasium as gym
    from omegaconf import OmegaConf
    import amago
    import envs
    from amago.envs import AMAGOEnv
    from amago.nets.tstep_encoders import FFTstepEncoder
    from amago.nets.traj_encoders import TformerTrajEncoder
    from amago.shielded_agent import ShieldedAMAGOAgent
    assert Path(amago.__file__).resolve().is_relative_to(ROOT)
    if name not in NAMES:
        raise ValueError("Unknown bundled checkpoint")
    relative_path = f"pretrained/{name}.pt"
    checkpoint = (ROOT / relative_path).resolve()
    if not checkpoint.is_relative_to(ROOT):
        raise ValueError("Checkpoint path escapes repository")
    manifest = json.loads((ROOT / "pretrained/manifest.json").read_text())
    entries = [entry for entry in manifest["files"] if entry["path"] == relative_path]
    if manifest.get("format") != 1 or len(entries) != 1:
        raise ValueError("Missing or ambiguous checkpoint manifest entry")
    entry = entries[0]
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if checkpoint.stat().st_size != entry["bytes"] or digest != entry["sha256"]:
        raise ValueError("Checkpoint size/SHA256 does not match manifest")
    torch.set_num_threads(1)
    torch.manual_seed(42)
    cfg = OmegaConf.load(ROOT / "pretrained" / f"{name}.yaml")
    ec = cfg.constrained_env_config
    if name == "darkroom":
        from envs.dark_room.envs.darkroom import CostWrapper
        base = gym.make(cfg.env_name, size=cfg.grid_size, goal_pos=(0, 0),
                        time_limit=cfg.time_limit, max_episodes=cfg.max_inner_episodes)
        base = CostWrapper(base, num=ec.cost_num, cost=ec.cost_cost, seed=cfg.seed,
                           min_cost_to_go=ec.min_cost_to_go, max_cost_to_go=ec.max_cost_to_go,
                           guranatee_path=ec.guranatee_path, reset_to_new_cost=ec.reset_to_new_cost,
                           radial_alpha=-ec.train_radial_alpha)
    else:
        import safety_gymnasium
        assert Path(safety_gymnasium.__file__).resolve().is_relative_to(ROOT)
        from envs.mujoco.mujoco import MujocoWrapper
        base = MujocoWrapper(cfg.env_name, seed=cfg.seed, time_limit=cfg.time_limit,
                             max_episodes=cfg.max_inner_episodes, min_cost_to_go=ec.min_ctg,
                             max_cost_to_go=ec.max_ctg, task_type=cfg.task_type,
                             sticky_movement_multiplier=cfg.sticky_movement_multiplier,
                             lidar=cfg.lidar, normalize_reward=cfg.normalize_reward,
                             radial_alpha=-ec.train_radial_alpha if cfg.task_type == "goal" else None)
    env = AMAGOEnv(base, env_name=cfg.env_name)
    try:
        timestep, _ = env.reset(seed=cfg.seed)
        obs, rl2, times = timestep.as_input()
        kwargs = OmegaConf.to_container(cfg.shield_config, resolve=True)
        kwargs["shield_log_margins"] = False
        policy = ShieldedAMAGOAgent(
            obs_space=env.observation_space,
            rl2_space=gym.spaces.Box(-np.inf, np.inf, shape=rl2.shape[1:], dtype=np.float32),
            action_space=env.action_space, max_seq_len=cfg.max_seq_len,
            tstep_encoder_type=FFTstepEncoder, traj_encoder_type=TformerTrajEncoder,
            constrained_config=cfg.constrained_agent_config,
            reward_multiplier=cfg.get("reward_multiplier", 10.0),
            cost_multiplier=cfg.get("cost_multiplier", None), **kwargs).cpu()
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if not isinstance(state, dict) or not all(isinstance(v, torch.Tensor) for v in state.values()):
            raise ValueError("Expected tensor-only policy state_dict")
        if not all(torch.isfinite(value).all().item() for value in state.values()):
            raise ValueError("Checkpoint contains nonfinite tensors")
        # All five bundled aliases match exactly: no legacy exceptions or random-head fallback.
        incompatible = policy.load_state_dict(state, strict=True)
        policy.eval()
        obs = {k: torch.from_numpy(v).unsqueeze(1) for k, v in obs.items()}
        with torch.no_grad():
            action, _ = policy.get_actions(obs=obs, rl2s=torch.from_numpy(rl2).unsqueeze(1),
                                          time_idxs=torch.from_numpy(times).unsqueeze(1),
                                          sample=cfg.get("sample_actions", False), hidden_state=None)
        if not torch.isfinite(action).all():
            raise ValueError("Nonfinite action")
        return {"benchmark": name, "status": "pass", "sha256": digest,
                "manifest_verified": True, "all_state_tensors_finite": True,
                "evaluation_context": cfg.max_seq_len, "action_shape": list(action.shape),
                "missing_keys": list(incompatible.missing_keys),
                "ignored_training_auxiliary_keys": list(incompatible.unexpected_keys),
                "environment_steps": 0}
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=NAMES)
    parser.add_argument("--output", type=Path, help="Optional JSON report path")
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--entrypoints", action="store_true", help="Also verify actual evaluation startup with evaluation replaced")
    args = parser.parse_args()
    if args.child:
        result = check_entrypoint(args.benchmark) if args.entrypoints else check(args.benchmark)
        print("CHECK_RESULT=" + json.dumps(result))
        return
    environment = os.environ.copy()
    environment.update(PYTHONPATH=os.pathsep.join(map(str, [ROOT / "amago", ROOT / "vendor/safety-gymnasium", ROOT])),
                       PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES="", TORCHDYNAMO_DISABLE="1",
                       OMP_NUM_THREADS="1", WANDB_MODE="disabled")
    results = []
    for name in ([args.benchmark] if args.benchmark else NAMES):
        command = [sys.executable, str(Path(__file__).resolve()), "--child", "--benchmark", name]
        proc = subprocess.run(command,
                              cwd=ROOT, env=environment, capture_output=True, text=True, timeout=180)
        if proc.returncode:
            print(proc.stdout, file=sys.stderr)
            print(proc.stderr, file=sys.stderr)
            raise SystemExit(proc.returncode)
        results.append(json.loads(next(line.removeprefix("CHECK_RESULT=") for line in proc.stdout.splitlines()
                                       if line.startswith("CHECK_RESULT="))))
        if args.entrypoints:
            # Separate from reset-inference check: benchmark patches are process-global.
            proc = subprocess.run(command + ["--entrypoints"], cwd=ROOT, env=environment,
                                  capture_output=True, text=True, timeout=180)
            if proc.returncode:
                print(proc.stdout, file=sys.stderr)
                print(proc.stderr, file=sys.stderr)
                raise SystemExit(proc.returncode)
            results[-1]["entrypoint_startup"] = json.loads(next(
                line.removeprefix("CHECK_RESULT=") for line in proc.stdout.splitlines()
                if line.startswith("CHECK_RESULT=")))
    report = json.dumps({"scope": "CPU state compatibility and reset observation only; no environment steps", "results": results}, indent=2)
    if args.output:
        args.output.write_text(report + "\n")
    print(report)


if __name__ == "__main__":
    main()
