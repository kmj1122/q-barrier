# Source release notes

This is a working-tree source snapshot with curated main-figure CSV exports
and five pretrained policy-weight aliases,
not an archived set of experiment configurations or a promise of bitwise
reproduction of every paper result. There are no replay buffers, optimizer
states, raw baseline data, raw result archives, private notes, cluster
launchers, or inherited Git history in this copy.

The modified AMAGO implementation, environment wrappers, training objectives,
and action-selection code are retained. Release preparation makes these changes:

- W&B defaults are off; project and evaluation run labels are generic.
- Evaluation epoch defaults are null, and a checkpoint-existence check runs
  before environment construction. Explicit checkpoint paths and the original
  run/epoch layout are supported; no random-weight evaluation fallback is added.
- The velocity training configuration explicitly declares the previously
  implicit `use_shield=false` option, so normal Hydra overrides work.
- Reviewer references and private debugging comments are removed without
  changing the shielding computation.
- Safety-Gymnasium's packaging permits Gymnasium0.28.1–0.29.x, consistent with
  the AMAGO requirement. Full-install requirements select Gymnasium0.29.0,
  MuJoCo2.3.3 and the legacy NumPy range required by Robotics1.2.2 metadata.
- Only assets needed by the five supported benchmarks are included. Other
  environment IDs registered by Safety-Gymnasium are not supported by this
  reduced asset distribution.
- Tests, offline release checks, and setup documentation are added. The project
  is MIT licensed; dependency notices and licenses remain intact.
- Five original `best_checkpoints` aliases are included unchanged in
  `pretrained/`, with exact bytes/SHA256, portable archived evaluation presets
  and explicit uncertainty about four original training identities. The
  checkpoint checker uses tensor-only loading and requires exact state keys.
- Optional training mode flags now respect their boolean values; previously
  merely defining `collect_only=false` could disable learning. Redirected
  MuJoCo resume now uses the actual policy/training-state checkpoint layouts.
  These two release-only entrypoint fixes have focused regression tests.

Scientific settings such as critic aggregation, uncertainty options, candidate
counts and budget-reset behavior are not silently rewritten to match a paper
description. Record explicit overrides; see `docs/CONFIGURATION.md`.

The two `data/figures/` CSVs preserve archived plotted values exactly; they
are not regenerated evaluations. Their `std` fields are shading widths, not
newly validated uncertainty estimates. See the data README, particularly
the warning about repeated observations in the historical RQ2 exporter.

`scripts/plot_main_figures.py` regenerates those figures directly from the
shipped CSVs using `requirements-plot.txt`. It requires no RL packages or
private research checkout. Numerical lines and shading widths match exactly;
small PDF rendering/layout differences are documented separately.
