# Pretrained policy checkpoints

Five existing policy-weight files are bundled unchanged (138,524,713 bytes in
total, approximately 132 MiB). They contain model tensors, not replay buffers,
optimizer/scheduler state, W&B credentials, or experiment logs.

| Checkpoint | Benchmark | Reported epoch | Evaluation context cap |
|---|---|---:|---:|
| `darkroom.pt` | DarkRoomHard-v0 | 2100 | 1500 |
| `point.pt` | SafetyPointGoal1-v0 | 2968 | 1500 |
| `car.pt` | SafetyCarGoal1-v0 | 2996 | 1500 |
| `ant.pt` | SafetyAntVelocity-v1 | 1834 | 2000 |
| `cheetah.pt` | SafetyHalfCheetahVelocity-v1 | 70 | 2000 |

The epoch column transcribes the historical checkpoint sidecar. Only DarkRoom
has been independently SHA256-matched to its original training-run epoch file.
The other four original training identities, seeds and trained context lengths
are not established by these aliases. In particular, similarly named Car and
HalfCheetah training files contain different weights and were not substituted.
The evaluation context caps in the table must not be presented as verified
training context lengths. See `manifest.json` for hashes, sizes and provenance.

The YAML presets use portable paths and archived evaluation settings: the
DarkRoom `shield_darkroom_later` evaluation and the continuous-task Q2 margin
diagnostics. They are not an assertion of exact main-figure reproduction.
They retain the historical budget semantics and critic aggregation rather
than silently changing them to match a paper description. See
[the configuration notes](../README.md#configuration).

## Check loading without running an evaluation

From the repository root, after installing the appropriate requirements:

```bash
python scripts/check_pretrained.py
# DarkRoom only; does not require MuJoCo:
python scripts/check_pretrained.py --benchmark darkroom
# Also exercise the production evaluation functions through model loading,
# replacing their evaluation loop with a tensor-equality check:
python scripts/check_pretrained.py --entrypoints
```

This CPU-only check verifies file integrity, strictly loads every model tensor,
initializes and resets the environment, and computes one action on the initial
observation. It takes **zero environment steps** and reports no benchmark
performance. Missing or incompatible weights fail; there is no random-weight
fallback. The checkpoint is never updated.

With `--entrypoints`, startup and the production checkpoint loaders run in
temporary output directories. The evaluation loop is replaced, not executed.
This invokes the underlying evaluation functions; Hydra CLI configuration
parsing is checked separately by `scripts/check_install.py`.

## Run an evaluation yourself

These commands run actual evaluations (not quick installation checks). No
training updates are performed, but simulator rollouts can be expensive.
Use the checkpoint-specific presets rather than assuming the generic training
or evaluation YAML describes the archived model.

```bash
python test_crl2_darkroom.py --config-path pretrained --config-name darkroom
python test_crl2_mujoco.py --config-path pretrained --config-name point
python test_crl2_mujoco.py --config-path pretrained --config-name car
python test_crl2_mujoco.py --config-path pretrained --config-name ant
python test_crl2_mujoco.py --config-path pretrained --config-name cheetah
```

Inspect settings with `--cfg job --resolve` first. DarkRoom's archived preset
uses 100 test goals and 150,000 steps **per actor**, so its default full
evaluation is substantial. For a deliberately smaller evaluation, override
`num_test_goals` and `val_timesteps_per_epoch` explicitly and report them.
Generated outputs go to `outputs/evaluation/`. Recording and W&B are disabled.

Policy-only checkpoints can initialize a new training run using
`load_checkpoint_path=./pretrained/<name>.pt`, with compatible settings. They
cannot exactly resume optimizer, scheduler or RNG state; that requires an
original full training-state checkpoint.
