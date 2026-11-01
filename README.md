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

This single requirements file includes training, all five environments, plotting,
and tests. NumPy is restricted to the legacy range required by
Gymnasium-Robotics 1.2.2; Matplotlib 3.9.2 preserves the archived figure backend.
Use a PyTorch 2.10.0 build appropriate for your hardware. FlashAttention is
optional; the included transformer has a standard attention fallback.
Headless rendering may require `MUJOCO_GL=egl` and system OpenGL drivers.
On a CPU-only host without graphics support, use `MUJOCO_GL=disable` for
non-rendering runs. Rendering support is platform-dependent.

For quick CPU checks, eager execution avoids compiler startup:

```bash
export TORCHDYNAMO_DISABLE=1
```

The initial release's compiled CPU stride failure was repaired by making
spectral-normalization buffers compact. The updated suite passes with CPU
compilation enabled in the checked development stack and in eager mode in a
fresh CPU installation. This flag changes compilation, not the learning
objective. See [verification](#verification) for the scope of the checks.

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

These start full training jobs, not quick tests. Inspect resolved settings first using `--cfg job --resolve`. See [configuration](#configuration) for budget semantics, context lengths, candidate sampling, and explicit uncertainty overrides. Output directories are controlled by `ckpt_base_dir` and `run_name`.

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

After installation, regenerate both figures directly from the CSVs:

```bash
python scripts/plot_main_figures.py
```

PDF and PNG files are written to `outputs/main_figures/`, together with
`reproduction_checks.json`. The script checks every plotted line and shading
coordinate against the CSV and does not apply a second scaling or smoothing.
Use `--figure RQ1` or `--figure RQ2` to regenerate only one figure.

The plotting script itself uses only NumPy and Matplotlib, not the RL modules,
checkpoints, or simulator. The unified installation includes the complete stack.

## Configuration

Hydra overrides use `key=value`. Inspect settings without starting a job:

```bash
python crl2_darkroom.py --cfg job --resolve use_shield=true
python crl2_mujoco.py --config-name crl2_velocity_config --cfg job --resolve use_shield=true env_name=SafetyAntVelocity-v1
```

The supplied configurations preserve their historical settings. They are
starting points, not an assertion that every archived experiment used them.
Save the resolved configuration alongside each run.

| Configuration family | Inner-episode limit | Inner episodes | Context cap |
|---|---:|---:|---:|
| DarkRoom | 30 steps | 50 | 1,500 steps |
| Point/Car goal | 75 policy decisions | 20 | 1,500 decisions |
| Ant/HalfCheetah velocity | 200 steps | 10 | 2,000 steps |

Point/Car repeat each policy action for up to five underlying environment
steps, stopping early at termination; these are not physics integration
substeps. `max_inner_episodes` changes the adaptation horizon and `time_limit`
changes the within-episode horizon. Some configs derive `max_seq_len` from
both, so override it explicitly when holding the context cap fixed. Always set
`env_name` explicitly for velocity tasks and match the checkpoint's task.

**Encoder-size limitation.** Top-level YAML fields `embedding_dim`,
`hidden_dim`, `num_layers`, and `num_heads` are legacy metadata, not wired to
the encoder constructors. Changing them alone does not resize the model. The
real CLI smoke runs used the effective defaults: a 256-wide, three-layer
transformer, verified from saved tensors. Inspect constructor/Gin settings and
checkpoint shapes. `max_seq_len` and supplied `shield_config` options are wired.

**Budgets.** The continuous-control wrapper resets the remaining budget at
each inner-episode boundary, retaining the sampled initial budget across
adaptation. DarkRoom's historical default does not reset at every inner
episode. To request per-inner-episode resets explicitly:

```bash
DARKROOM_RESET_BUDGET_PER_INNER_EPISODE=1 python crl2_darkroom.py use_shield=true
```

Use the same setting during evaluation. This changes the task protocol, not
just a diagnostic, and should be reported.

**Shield and sampling.** `use_shield=true` selects the shielded agent. Discrete
actions are scored directly; continuous candidate counts use
`shield_config.shield_continuous_num_samples`. The supplied velocity training
config uses one candidate and evaluation uses eight. A single candidate cannot
be replaced by reweighting. Point/Car train and evaluation configs both use
eight. Preserve fast-path/staged-sampling settings when comparing runs.
Continuous training requests stochastic `sample_actions`, while evaluation
configs set it to false. Record these choices with results.

DarkRoom's training YAML retains an uncertainty option. Disable it explicitly
when intended, rather than assuming it is absent:

```bash
python crl2_darkroom.py use_shield=true shield_config.shield_uncertainty_mode=none shield_config.shield_uncertainty_kappa=0.0
```

Hydra requires `+` when adding fields absent from a configuration, for example
`+shield_config.shield_uncertainty_mode=none` in some evaluation YAMLs.

**Checkpoints.** Match the environment, encoder dimensions, critic count,
auxiliary modules and protocol when loading weights. `load_checkpoint_path`
must identify an existing compatible policy; there is no random-weight fallback.
Only DarkRoom's original training identity was independently hash-matched.
The other presets' context caps describe evaluation, not verified training
context lengths. See the [checkpoint notes](pretrained/README.md). Output
locations use `ckpt_base_dir` and `run_name`; W&B is optional via `use_wandb=true`.
Optional collection/learning flags respect their boolean values, and redirected
MuJoCo resume uses the same policy/full-state layouts as checkpoint saving.

## Verification

The unified requirements installed successfully in the isolated Python 3.10
CPU environment, with `pip check` reporting no conflicts. The final suite
passed **72 tests** there in eager mode and **72 tests** with CPU compilation
enabled in the development environment. Both main figures also regenerated
directly from the bundled CSVs.

```bash
python -m pip check
python scripts/check_release.py --selftest
TORCHDYNAMO_DISABLE=1 python -m pytest tests/
# Strict model loading, initial-observation action, and actual evaluation startup:
# No environment steps, benchmark evaluations, or training updates.
python scripts/check_pretrained.py --entrypoints
python scripts/check_install.py --mujoco
```

The pretrained checker verifies hashes, exact model loading and an initial
action. With `--entrypoints`, it exercises production startup but replaces
the evaluation loop with a tensor-equality check: it takes zero environment
steps. The unit suite separately includes finite optimizer/save/load/cache
checks on synthetic data and optional GIF rendering.

Additional bounded checks on 2026-11-01 exercised real production rollouts for
all five pretrained policies (two inner episodes each), real short DarkRoom
and Point training, full-state DarkRoom restart, and policy-only Point
continuation. Encoder/dynamics weights changed, saved metrics were finite, and
pretrained hashes stayed unchanged. A fresh isolated Python 3.10 CPU environment
also passed real DarkRoom/Point evaluation and Point training. RTX 5080 checks
passed eager DarkRoom training and compiled discrete/continuous optimizer
updates; an initial cold-compile timeout was retained, and the successful
follow-up reused compiler cache. GPU checks preceded the local restore fixes;
those fixes were verified on CPU.

The checks exposed and repaired compact-buffer checkpoint/CPU-compilation
failures, legacy NumPy RNG restoration during full-state restart, and an
omitted optional GIF helper. No unrestricted deserialization fallback was
added. Full-state regression tests verify model, optimizer, scheduler, and
Python/NumPy/Torch CPU RNG restoration plus a subsequent update. The temporary
legacy dtype allowance preserves caller permissions but is process-global,
not thread-local. The bundled policies contain no optimizer/scheduler state.

These are software checks, not reproduction of paper scores or safety
certificates. Full-scale convergence, long-horizon adaptation, multi-GPU,
asynchronous rollout performance, bitwise environment-state restart and
cross-platform rendering remain unverified. Evaluation uncertainty is not
variability across independent training seeds. The [CSV notes](data/figures/README.md)
explain additional limitations of the archived shading widths.

## Licensing

Original project additions are licensed under the [MIT License](LICENSE).
Bundled third-party code retains its original licenses and copyright headers:

- **AMAGO:** `amago/`, from [UT-Austin-RPL/amago](https://github.com/UT-Austin-RPL/amago).
  Copyright Jake Grigsby and upstream contributors; [MIT license](amago/LICENSE).
  This is a modified copy with constrained learning, shielding and compatibility
  changes, not an unmodified upstream release.
- **OmniSafe Lagrange utility:** `amago/amago/lagrange.py` is an exception to
  AMAGO's MIT license. Copyright 2023 OmniSafe Team; its original Apache-2.0
  header is retained. The full [Apache-2.0 license](vendor/safety-gymnasium/LICENSE)
  is bundled.
- **Safety-Gymnasium:** `vendor/safety-gymnasium/`, from
  [PKU-Alignment/safety-gymnasium](https://github.com/PKU-Alignment/safety-gymnasium).
  Copyright OmniSafe Team and upstream contributors;
  [Apache-2.0 license](vendor/safety-gymnasium/LICENSE) and file notices retained.
  This copy includes project-specific velocity environments, packaging
  compatibility changes and only the assets needed by the five supported
  benchmarks. Other registered environments may require omitted assets.

Other installed dependencies retain their own licenses. SCARED curves derive
from experimental data supplied by its authors; only method-attributed plotting
summaries are included, not raw external archives. These summaries do not
relicense the underlying third-party data; see [data provenance](data/figures/README.md).
