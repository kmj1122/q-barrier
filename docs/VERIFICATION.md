# Local release verification

Initial source/figure checks were performed on 2026-10-29 without training,
optimizer updates, policy evaluation, benchmark result generation, commits,
or network publication. The 2026-10-30 additions below use finite synthetic
optimizer tests and unchanged pretrained checkpoints, not new experiments.

## Public-release recheck (2026-11-01)

- Release-checker self-tests: **10/10 passed**; source/layout checks and all
  source-manifest and checkpoint hashes pass.
- The default compiled CPU unit suite reports **56 passed, 3 failed**. The
  failures are TorchInductor stride assertions in compiled attention.
- In the same environment, `TORCHDYNAMO_DISABLE=1 python -m pytest tests/ -q`
  reports **59 passed in 20.85 seconds**. This is the documented CPU workaround;
  no learning code or scientific settings were changed.
- This recheck does not cover a fresh full installation or GPU execution.
  The dated results below describe earlier checks, not an assertion that the
  default compiled CPU command passes in the current environment.

## Training pipeline and bundled checkpoints (2026-10-30)

- Full unit suite: **59 passed in 19.57 seconds** in the checked CPU environment.
  This includes six new pipeline smoke checks, eighteen training-entrypoint
  regressions and eight checkpoint-integrity checks.
- Synthetic training checks use the real `Experiment.init_model` and
  `Experiment.train_step` for discrete and continuous actions. They verify
  finite losses/parameters, one optimizer update, scheduler and gradient-counter
  advancement, target synchronization, changed online encoder/dynamics weights,
  valid padding masks and unchanged frozen RND parameters.
- Synthetic checkpoint tests use the production save/load helpers and require
  exact parameter/action agreement after a round trip. Missing backbone keys
  and incompatible critic counts are rejected.
- Cached synthetic inference goes past an eight-step context cap and verifies
  exact cache-length progression/saturation, finite active cache entries and
  selective reset parity. This checks software mechanics, not a claim of
  full-window recomputation equivalence or proven memory retention.
- Five policy checkpoints are copied unchanged into `pretrained/`. Their
  sizes and SHA256 match the existing release aliases. Loading is tensor-only,
  all state tensors are finite, strict state matching has zero missing or
  ignored parameters, and an initial-observation action is finite for each
  benchmark. These checks take **zero environment steps** and never update
  the bundled models.
- All five production evaluation functions pass startup and their real
  checkpoint loaders preserve every saved tensor exactly. The evaluation
  loop is replaced by the comparison callback. Separately, all eleven Hydra
  CLI configurations resolve (six generic train/evaluation selections plus
  five checkpoint presets), and all four continuous benchmark assets reset.
- Only DarkRoom was hash-matched to an original training epoch file.
  Epochs for Point, Car, Ant and HalfCheetah remain historical sidecar claims;
  their original training identities and context lengths are not independently
  established. Evaluation presets must not be mistaken for training metadata.
- Fixed two optional entrypoint bugs in the clean copy: boolean-valued
  collection/learning flags now respect false values, and redirected MuJoCo
  resume uses the same policy/training-state paths as the save helpers.
- A separate independent audit found no major/critical issue. Minor findings
  led to updated checkpoint documentation and stronger cache-progression
  assertions.

Re-run bounded verification from the repository root:

```bash
python -m pytest tests/
python scripts/check_install.py --mujoco
python scripts/check_pretrained.py --entrypoints
python scripts/check_release.py --selftest
```

The checkpoint checker's `--entrypoints` option runs the underlying production
evaluation functions through initialization and checkpoint loading, then
**replaces** their evaluation loop with an exact tensor-equality check. Output
files go to temporary directories. It does not run a benchmark evaluation.
Hydra CLI configuration composition is checked separately by `check_install.py`.
The local machine-readable checkpoint report is generated in
`outputs/verification/pretrained_checks.json` (ignored by Git).

## Completed checks

- Initial core unit tests: **17 passed**, including random-weight shielding tests,
  mocked environment reset behavior, missing-checkpoint rejection and
  release-checker failure fixtures.
- Main-figure CSV addition: **4 targeted tests passed**, covering the two
  new CSV/hash/schema checks and the two existing release-check tests.
  Both CSVs are byte-identical to the archived exports. The corresponding
  paper PDFs match the archived PDFs after removing creation-date metadata.
- CSV-only rendering: **12 targeted tests passed** (rendering regressions,
  CSV integrity and release checks). Both figures also regenerated in a
  fresh environment installed from `requirements-plot.txt`, with no Torch,
  AMAGO, Safety-Gymnasium or pandas installed. A separate package containing
  only the plotting script and two CSVs produced the same PDF content after
  removing creation-date metadata.
- Rendering fidelity: all1,440 plotted points and100 shading polygons match
  the CSV coordinates exactly. PDF labels match the paper exactly; maximum
  text-coordinate differences are0.637points for RQ1 and0.610points for RQ2.
  Regenerated PDFs are visually equivalent, not byte-identical to the paper.
- Release checker: **10/10 self-test cases passed** and the source/layout
  scan passed. This is a bounded heuristic scan, not a universal secret detector.
- Configuration loading: all six train/evaluation configuration selections
  load and resolve with `use_shield=true`.
- Dependency isolation: AMAGO, local wrappers and Safety-Gymnasium imports
  resolve to this clean repository, not the original research checkout.
- Asset checks: PointGoal1, CarGoal1, AntVelocity-v1 and
  HalfCheetahVelocity-v1 construct and reset with finite initial observations.
  No actions or trained policies are used in this check.
- Dependency resolution: the Python3.10 continuous-control requirements
  resolve successfully after reconciling the legacy Robotics NumPy constraint.
- Source integrity: copied implementation files retain their Python AST
  after ignoring docstrings, except the documented evaluation-path guard and
  optional training-entrypoint fixes. See `SOURCE_MANIFEST.json`.

## Validation limits

Runtime checks used an existing development environment: Python3.10.19,
PyTorch2.10.0, NumPy2.2.6, Gymnasium0.29.0 and MuJoCo2.3.3. The resolver
selects NumPy1.23.5 for a new continuous-control installation because
Gymnasium-Robotics1.2.2 declares NumPy<1.24. That complete fresh environment
has not yet been installed and executed. Resolution is not an installation test.

The CSV-only plotting environment is separate and was installed and executed:
Python3.10.19, Matplotlib3.9.2 and NumPy2.2.6, plus Matplotlib's dependencies.
It does not require the full RL installation described in the preceding paragraph.

These checks establish source-package and entrypoint usability in the checked
environment, not reproduction of published returns/costs, performance on
every platform, or formal safety of the learned shield.

No full training run, optimizer-state resume, long benchmark rollout or new
scientific evaluation was performed. GPU execution, distributed training,
asynchronous rollout performance and a fresh full RL installation remain
outside this verification scope. The included policy-only weights cannot
resume original optimizer/scheduler/RNG state exactly.

The main-figure data are archived plotting outputs, not a fresh validation of
statistical uncertainty. In particular, the historical RQ2 calculation repeats
trajectory summaries before grouping. Its preserved shading widths should
not be described as independently validated rollout SEM; see `data/figures/README.md`.
