# Configuration and protocol

Hydra overrides take the form `key=value`. Inspect a configuration before launching work:

```bash
python crl2_darkroom.py --cfg job --resolve use_shield=true
python crl2_mujoco.py --config-name crl2_velocity_config --cfg job --resolve use_shield=true env_name=SafetyAntVelocity-v1
```

The copied configurations preserve their scientific settings. They are starting configurations, not a claim that every archived experiment used identical settings. Save the resolved configuration for each run.

## Environments and horizons

| Config family | Inner-episode limit | Inner episodes | Context cap |
|---|---:|---:|---:|
| DarkRoom | 30 steps | 50 | 1,500 steps |
| Point/Car goal | 75 policy decisions | 20 | 1,500 decisions |
| Ant/HalfCheetah velocity | 200 steps | 10 | 2,000 steps |

Point/Car repeat each policy action for up to five underlying environment steps, stopping early at termination. This is not the same as MuJoCo physics integration substeps. Increasing `max_inner_episodes` changes the evaluation adaptation horizon, while increasing `time_limit` changes the within-episode horizon. Several configs derive `max_seq_len` from both fields, so override `max_seq_len` explicitly when the intended experiment holds the context cap fixed.

The velocity training and evaluation configs have different default environment names. Always set `env_name` explicitly and match the checkpoint's training task.

## Budget semantics

The continuous-control wrapper resets the remaining budget at each inner-episode reset and retains the sampled initial budget within the adaptation trajectory.

DarkRoom preserves a historical default: its cost wrapper does not reset the remaining budget at every inner-episode boundary. To request per-inner-episode budget resets, set the environment variable explicitly for both training and evaluation:

```bash
DARKROOM_RESET_BUDGET_PER_INNER_EPISODE=1 python crl2_darkroom.py use_shield=true
```

Leaving the variable unset preserves the historical behavior. Changing it changes the task protocol and should be reported; it is not merely a plotting option. Match this setting when training and evaluating a checkpoint.

## Shield and sampling

Set `use_shield=true` to select the shielded agent. In discrete environments the shield scores the available discrete actions. Continuous candidates are controlled by `shield_config.shield_continuous_num_samples`.

The supplied velocity training config uses one candidate, while its evaluation config uses eight. With a single candidate, reweighting cannot choose an alternative action. The supplied Point/Car training and evaluation configs both specify eight candidates, so do not describe all continuous training configurations as using one. Fast-path and staged-sampling settings can also affect candidate evaluation; preserve them when comparing runs.

`sample_actions` controls stochastic policy action sampling. The continuous-control training configs request stochastic sampling, while evaluation configs specify `sample_actions=false`. This choice and candidate-set settings should accompany reported results.

DarkRoom's training YAML retains an uncertainty option. To disable it explicitly:

```bash
python crl2_darkroom.py use_shield=true shield_config.shield_uncertainty_mode=none shield_config.shield_uncertainty_kappa=0.0
```

Some evaluation YAMLs do not declare these fields. Hydra requires `+` when adding absent fields, for example `+shield_config.shield_uncertainty_mode=none`. Inspect the resolved config and match the trained model rather than assuming all variants are interchangeable.

## Checkpoints and outputs

Evaluation requires `load_checkpoint_path` to identify an existing compatible policy checkpoint. The generic `load_checkpoint_epoch` default is not evidence of a supplied checkpoint's training epoch. Match encoder dimensions, critic count, auxiliary modules, environment, and protocol when loading weights.

`ckpt_base_dir` controls output storage. Logging to W&B is optional through `use_wandb=true`. Rendering and GIF/video generation may require additional system graphics support. Five existing pretrained policy aliases and compatible evaluation presets are distributed in `pretrained/`; raw external baseline archives are not. Only DarkRoom's original training identity was independently verified by an exact checkpoint hash match. The other presets' context caps describe evaluation, not verified training context lengths. The aggregate main-figure CSVs in `data/figures/` preserve archived plotting outputs and documented uncertainty limitations.
