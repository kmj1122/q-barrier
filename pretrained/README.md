# Pretrained policies

This folder has one pretrained Q-Barrier policy per benchmark (about 132 MiB in total). Each file holds the model weights only, without replay buffers or optimizer state. `manifest.json` lists the file hashes.

| Checkpoint | Benchmark | Context length |
|---|---|---:|
| `darkroom.pt` | DarkRoomHard-v0 | 1,500 |
| `point.pt` | SafetyPointGoal1-v0 | 1,500 |
| `car.pt` | SafetyCarGoal1-v0 | 1,500 |
| `ant.pt` | SafetyAntVelocity-v1 | 2,000 |
| `cheetah.pt` | SafetyHalfCheetahVelocity-v1 | 2,000 |

Each `.yaml` file in this folder is the evaluation config for the checkpoint of the same name.

## Check that the checkpoints load

From the repository root:

```bash
python scripts/check_pretrained.py                       # all five
python scripts/check_pretrained.py --benchmark darkroom  # DarkRoom only, no MuJoCo needed
```

This loads each model, checks its hash and computes one action. It does not run an evaluation.

## Evaluate

```bash
python test_crl2_darkroom.py --config-path pretrained --config-name darkroom
python test_crl2_mujoco.py --config-path pretrained --config-name point
python test_crl2_mujoco.py --config-path pretrained --config-name car
python test_crl2_mujoco.py --config-path pretrained --config-name ant
python test_crl2_mujoco.py --config-path pretrained --config-name cheetah
```

Results are written to `outputs/evaluation/`. The DarkRoom config evaluates 100 test goals with 150,000 steps per actor. For a shorter run, lower `num_test_goals` and `val_timesteps_per_epoch`.

## Fine-tune

To start a new training run from a checkpoint, pass `load_checkpoint_path=./pretrained/<name>.pt` with the same environment and model settings.
