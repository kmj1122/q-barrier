# Q-Barrier

Code for **Latent Q-Barrier Shielding for Safe In-Context Reinforcement Learning** (NeurIPS 2026). Policies adapt through interaction history with frozen parameters at evaluation time. The learned barrier reweights predicted risk; it does not certify closed-loop safety.

Supported benchmarks are **DarkRoomHard-v0**, **SafetyPointGoal1-v0**, **SafetyCarGoal1-v0**, **SafetyAntVelocity-v1**, and **SafetyHalfCheetahVelocity-v1**. The bundled Safety-Gymnasium assets cover these benchmarks only. Five existing [pretrained policy checkpoints](pretrained/README.md), evaluation presets, and aggregate plotting values for the two main figures are included. Raw experiment archives are not distributed.

## Installation

Python 3.10 is recommended. From this directory:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For Point, Car, Ant, and HalfCheetah, additionally install:

```bash
python -m pip install -r requirements-mujoco.txt
```

The core dependency versions follow the checked development stack. The continuous-control requirements additionally use the legacy NumPy range required by Gymnasium-Robotics 1.2.2. Install a PyTorch 2.10.0 build appropriate for your hardware. FlashAttention is optional; the included transformer has a standard attention fallback. Headless MuJoCo rendering may require `MUJOCO_GL=egl` and system OpenGL drivers. These instructions have not yet been validated in a newly created environment on every supported platform.

For CPU use in the checked PyTorch 2.10.0 environment, disable compilation:

```bash
export TORCHDYNAMO_DISABLE=1
```

The compiled CPU attention path can raise a TorchInductor stride assertion.
Eager CPU execution passes the unit suite; this setting changes compilation,
not the model or learning objective. See [verification details](docs/VERIFICATION.md).

## Training

Commands below explicitly enable the shield; the preserved YAML configurations also support `use_shield=false` for the base constrained agent. W&B logging is disabled by default.

```bash
# DarkRoom
python crl2_darkroom.py use_shield=true

# Goal navigation: replace Point with Car for SafetyCarGoal1-v0
python crl2_mujoco.py use_shield=true env_name=SafetyPointGoal1-v0

# Velocity: replace Ant with HalfCheetah for SafetyHalfCheetahVelocity-v1
python crl2_mujoco.py --config-name crl2_velocity_config use_shield=true env_name=SafetyAntVelocity-v1
```

These start full training jobs, not quick tests. Inspect resolved settings first using `--cfg job --resolve`. See [configuration notes](docs/CONFIGURATION.md) for budget semantics, context lengths, candidate sampling, and explicit uncertainty overrides. Output directories are controlled by `ckpt_base_dir` and `run_name`.

## Evaluation

Use the bundled checkpoint-specific evaluation presets:

```bash
python test_crl2_darkroom.py --config-path pretrained --config-name darkroom
# Use --config-name car for SafetyCarGoal1-v0:
python test_crl2_mujoco.py --config-path pretrained --config-name point
# Use --config-name cheetah for SafetyHalfCheetahVelocity-v1:
python test_crl2_mujoco.py --config-path pretrained --config-name ant
```

These run real evaluations, not quick tests. See the [checkpoint notes](pretrained/README.md) for sizes, provenance limitations and runtime settings. In particular, the archived DarkRoom preset uses 150,000 evaluation steps per actor. To use your own weights, pass `load_checkpoint_path` and match the environment, architecture and shield settings. Evaluation and training sampling differ; record overrides alongside results. The bundled presets are not a claim of exact main-figure reproduction.

## Repository layout

| Path | Contents |
|---|---|
| `crl2_*.py`, `test_crl2_*.py` | Training and evaluation entrypoints |
| `*_config.yaml` | Hydra configurations; velocity uses the MuJoCo entrypoints |
| `amago/` | Modified AMAGO learning, transformer, and shielding implementation |
| `envs/` | DarkRoom and continuous-control wrappers |
| `utils/` | Goal generation, seeding, and optional recording helpers |
| `vendor/safety-gymnasium/` | Safety-Gymnasium source and benchmark-specific assets |
| `data/figures/` | Archived plotted values for the two main results figures |
| `pretrained/` | Five unchanged policy checkpoints, integrity manifest and evaluation presets |
| `tests/`, `scripts/check_release.py` | Unit tests and structural release checks |

## Main-figure CSVs

- [RQ1: adaptation curves](data/figures/RQ1_aggregated_plot_values.csv)
- [RQ2: budget sweeps](data/figures/RQ2_aggregated_plot_values.csv)

These preserve the plotted values, uncertainty widths, and scaling notes.
Read the [data notes](data/figures/README.md) before reuse: the legacy `std`
column records the original shading width and is not automatically a
validated rollout-level SEM, particularly for the historical RQ2 exporter.

Regenerate both figures using only those CSVs and the plotting dependencies
(no training, checkpoints, simulator, or RL packages required):

```bash
python -m pip install -r requirements-plot.txt
python scripts/plot_main_figures.py
```

PDF and PNG files are written to `outputs/main_figures/`, together with
`reproduction_checks.json`. The script checks every plotted line and shading
coordinate against the CSV and does not apply a second scaling or smoothing.
Use `--figure RQ1` or `--figure RQ2` to regenerate only one figure.

## Verification

```bash
python -m pip install -r requirements-dev.txt
python scripts/check_release.py
TORCHDYNAMO_DISABLE=1 python -m pytest tests/
# Strict model loading, initial-observation action, and actual evaluation startup:
# No environment steps, benchmark evaluations, or training updates.
python scripts/check_pretrained.py --entrypoints
# After installing requirements-mujoco.txt:
python scripts/check_install.py --mujoco
```

The unit tests include finite optimizer/save/load/cache tests on synthetic data; pretrained checks do not update the bundled weights. These checks do not establish experimental performance or replace full training/evaluation. See [verification details](docs/VERIFICATION.md). Error bars from evaluation trajectories should not be described as variability across independent training seeds.

## Licensing

Original project additions are licensed under the [MIT License](LICENSE). Bundled third-party source retains its own licenses; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md), including the Apache-2.0 terms for Safety-Gymnasium.
