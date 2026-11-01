"""Check configuration loading and optional MuJoCo asset initialization.

No trained policies, optimizer steps, benchmark rollouts, or experiment results.
The optional asset check only constructs and resets environments, without actions.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
CONFIGS = (
    ("crl2_darkroom.py", []),
    ("crl2_mujoco.py", []),
    ("crl2_mujoco.py", ["--config-name", "crl2_velocity_config"]),
    ("test_crl2_darkroom.py", []),
    ("test_crl2_mujoco.py", []),
    ("test_crl2_mujoco.py", ["--config-name", "test_crl2_velocity_config"]),
    ("test_crl2_darkroom.py", ["--config-path", "pretrained", "--config-name", "darkroom"]),
    *(("test_crl2_mujoco.py", ["--config-path", "pretrained", "--config-name", name])
      for name in ("point", "car", "ant", "cheetah")),
)


def child_environment() -> dict:
    environment = os.environ.copy()
    local_paths = [str(ROOT / "amago"), str(ROOT / "vendor/safety-gymnasium"), str(ROOT)]
    if environment.get("PYTHONPATH"):
        local_paths.append(environment["PYTHONPATH"])
    environment.update({
        "PYTHONPATH": os.pathsep.join(local_paths),
        "PYTHONDONTWRITEBYTECODE": "1",
        "CUDA_VISIBLE_DEVICES": "",
        "TORCHDYNAMO_DISABLE": "1",
        "OMP_NUM_THREADS": "1",
        "WANDB_MODE": "disabled",
    })
    return environment


def run(command: list[str], environment: dict) -> None:
    result = subprocess.run(command, cwd=ROOT, env=environment, capture_output=True, text=True, timeout=180)
    if result.returncode:
        print(result.stdout)
        print(result.stderr, file=sys.stderr)
        raise SystemExit(result.returncode)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mujoco", action="store_true", help="Also initialize the four continuous benchmarks")
    args = parser.parse_args()
    environment = child_environment()
    for entrypoint, options in CONFIGS:
        run([sys.executable, entrypoint, *options, "--cfg", "job", "--resolve", "use_shield=true"], environment)
        print(f"PASS configuration: {entrypoint} {' '.join(options)}", flush=True)

    if args.mujoco:
        # Fresh processes avoid stacking the original benchmark's global patches.
        for name, task_type in (
            ("SafetyPointGoal1-v0", "goal"), ("SafetyCarGoal1-v0", "goal"),
            ("SafetyAntVelocity-v1", "velocity"), ("SafetyHalfCheetahVelocity-v1", "velocity"),
        ):
            program = (
                "from pathlib import Path\n"
                "import numpy as np\n"
                "import safety_gymnasium\n"
                "assert Path(safety_gymnasium.__file__).resolve().is_relative_to(Path.cwd())\n"
                "from envs.mujoco.mujoco import MujocoWrapper\n"
                f"env = MujocoWrapper({name!r}, seed=42, time_limit=75, max_episodes=2, "
                f"min_cost_to_go=10, max_cost_to_go=50, task_type={task_type!r}, "
                "sticky_movement_multiplier=5 if " + repr(task_type) + " == 'goal' else 1, lidar=False)\n"
                "try:\n"
                "    obs, info = env.reset(seed=42)\n"
                "    assert np.isfinite(obs['observation']).all()\n"
                "    assert np.isfinite(obs['cost_to_go']).all()\n"
                "finally:\n"
                "    env.close()\n"
            )
            run([sys.executable, "-c", program], environment)
            print(f"PASS assets and initialization: {name}", flush=True)
    print("Installation checks passed; no policy evaluation or training was performed.")


if __name__ == "__main__":
    main()
