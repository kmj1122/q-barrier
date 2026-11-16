# Q-Barrier

Code for **Latent Q-Barrier Shielding for Safe In-Context Reinforcement Learning** (NeurIPS 2026).

The repository supports five benchmarks: **DarkRoomHard-v0**, **SafetyPointGoal1-v0**, **SafetyCarGoal1-v0**, **SafetyAntVelocity-v1** and **SafetyHalfCheetahVelocity-v1**. It includes [pretrained policies](pretrained/README.md) for all five, their evaluation configs, and the plotted values of the two main figures.

## Installation

Python 3.10 is recommended.

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Install a PyTorch 2.10.0 build that matches your hardware. FlashAttention is optional. For headless rendering, set `MUJOCO_GL=egl`.

## Training

```bash
# DarkRoom
python crl2_darkroom.py use_shield=true

# Goal navigation (use SafetyCarGoal1-v0 for Car)
python crl2_mujoco.py use_shield=true env_name=SafetyPointGoal1-v0

# Velocity (use SafetyHalfCheetahVelocity-v1 for HalfCheetah)
python crl2_mujoco.py --config-name crl2_velocity_config use_shield=true env_name=SafetyAntVelocity-v1
```

Set `use_shield=false` to train the base agent without the shield. `ckpt_base_dir` and `run_name` set the output location. W&B logging is off by default and can be turned on with `use_wandb=true`.

## Evaluation

```bash
python test_crl2_darkroom.py --config-path pretrained --config-name darkroom
python test_crl2_mujoco.py --config-path pretrained --config-name point   # or car
python test_crl2_mujoco.py --config-path pretrained --config-name ant     # or cheetah
```

To evaluate your own weights, pass `load_checkpoint_path` and use the same environment, architecture and shield settings as in training.

## Main figures

The plotted values of the two main figures are in [`data/figures/`](data/figures/README.md):

- [RQ1: adaptation curves](data/figures/RQ1_aggregated_plot_values.csv)
- [RQ2: budget sweeps](data/figures/RQ2_aggregated_plot_values.csv)

To redraw both figures from these files:

```bash
python scripts/plot_main_figures.py              # add --figure RQ1 or --figure RQ2 for one figure
```

The figures are written to `outputs/main_figures/`.

## Configuration

Settings are Hydra overrides of the form `key=value`. To print the resolved config without running:

```bash
python crl2_darkroom.py --cfg job --resolve use_shield=true
```

| Environment | Steps per episode | Episodes per context | Context length |
|---|---:|---:|---:|
| DarkRoom | 30 | 50 | 1,500 |
| Point/Car | 75 | 20 | 1,500 |
| Ant/HalfCheetah | 200 | 10 | 2,000 |

- **Context.** `max_inner_episodes` and `time_limit` set the number of episodes and the episode length. `max_seq_len` is derived from both.
- **Model size.** The model size is set by the constructor defaults, not by the top-level `embedding_dim`, `hidden_dim`, `num_layers` and `num_heads` fields.
- **Budget.** In Point, Car, Ant and HalfCheetah, the remaining budget resets at every episode. In DarkRoom it does not reset by default. Set `DARKROOM_RESET_BUDGET_PER_INNER_EPISODE=1` to reset it, and use the same setting for training and evaluation.
- **Candidates.** In continuous-action tasks, `shield_config.shield_continuous_num_samples` sets the number of candidate actions. Velocity training uses 1 and evaluation uses 8. Point and Car use 8 in both.

## Repository layout

| Path | Contents |
|---|---|
| `crl2_*.py`, `test_crl2_*.py` | Training and evaluation scripts |
| `*_config.yaml` | Hydra configs |
| `amago/` | AMAGO with the constrained learning and shielding code |
| `envs/` | DarkRoom and MuJoCo wrappers |
| `vendor/safety-gymnasium/` | Safety-Gymnasium with the velocity tasks |
| `pretrained/` | Pretrained policies and their evaluation configs |
| `data/figures/` | Plotted values of the main figures |
| `tests/`, `scripts/` | Tests and helper scripts |

## Tests

```bash
TORCHDYNAMO_DISABLE=1 python -m pytest tests/
python scripts/check_pretrained.py
```

## License

Our code is released under the [MIT License](LICENSE). Bundled third-party code keeps its own license:

- `amago/` is a modified copy of [AMAGO](https://github.com/UT-Austin-RPL/amago) ([MIT](amago/LICENSE)).
- `amago/amago/lagrange.py` is from OmniSafe ([Apache-2.0](vendor/safety-gymnasium/LICENSE)).
- `vendor/safety-gymnasium/` is a modified copy of [Safety-Gymnasium](https://github.com/PKU-Alignment/safety-gymnasium) ([Apache-2.0](vendor/safety-gymnasium/LICENSE)).
