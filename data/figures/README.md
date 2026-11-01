# Main-figure plotting data

These are byte-for-byte copies of the archived **plot-ready values**, not
raw evaluation trajectories or newly computed experiment results.

| File | Figure | Rows | Curves |
|---|---|---:|---:|
| `RQ1_aggregated_plot_values.csv` | Return/cost versus adaptation episode | 1,000 | 50 |
| `RQ2_aggregated_plot_values.csv` | Return/cost versus cost budget | 440 | 50 |

Each file covers five environments and five methods: Q-Barrier, SCARED,
Safe AD, MAML with penalty, and Safe Meta. Each environment/method has
separate return and cost curves. Original algorithm and environment aliases
are retained to preserve the original export schema.

## Columns

- `rq`: RQ1 or RQ2.
- `environment`, `environment_display`: original environment key and panel label.
- `algorithm`, `algorithm_display`: original method key and legend label.
- `metric`, `metric_label`: return/cost key and the original axis label.
- `x_name`, `x_value`: the coordinate actually passed to the plotting code.
- `mean`: the plotted curve value, after the original transformations.
- `std`: the archived shading half-width, used to draw `mean ± std`.
- `source_mean_column`, `source_std_column`: column names in the input tables.
- `notes`: plot-time scale factors, already applied to `mean` and `std`.

**Do not reapply scaling or smoothing.** The values already include those
operations. The `notes` field is retained unchanged, including the baseline
scale factors; reproducing these exports does not independently validate the
scientific justification of each historical normalization.

RQ1 preserves the original **zero-based** episode coordinates: DarkRoom
has indices0–49, Point/Car0–14, and Ant/HalfCheetah0–9. For one-based display,
add1 deliberately rather than assuming the stored index starts at1.
RQ2 preserves the plotted budget coordinates. Most curves have9 points;
the two Safe AD DarkRoom curves have4. Missing budget points are not filled.
Despite legacy source-column names such as `mean_max_cost`, the inspected
RQ2 producer computes mean episodic cost, not a maximum or95th percentile.

## Uncertainty limitations

The legacy name `std` does **not** establish that a value is a raw sample
standard deviation or uncertainty across independent training seeds.
The inspected RQ1 exporter computes evaluation SEM and then smooths it.
The inspected RQ2 exporter repeats each trajectory summary across its episode
budget entries before grouping and calculating a nominal standard error.
Repeated entries are not independent rollouts. Those archived RQ2 widths are
preserved for figure fidelity, not endorsed as independently validated SEM.
Recompute uncertainty from the appropriate independent units for new inference.

These tables do not contain raw samples, training-seed replicates, or the
identifiers needed for a new goal-cluster analysis. No new significance or
calibration claim is supported by repackaging them.

## Provenance and attribution

The original `plot_rq1.py` and `plot_rq2.py` exporters recorded the same
transformed arrays passed to the line and shading calls. Their output PDFs
match the paper's corresponding PDF assets byte-for-byte after removing only
the PDF `/CreationDate` metadata field. File hashes and the comparison record
are in `provenance.json`.

SCARED curves use experimental data provided by the SCARED authors; this
release does not claim independent retraining of that baseline. These are
derived plotting summaries with method attribution, not a redistribution or
relicensing of the underlying third-party experiment archives.

## Regenerating the figures

From the repository root:

```bash
python -m pip install -r requirements.txt
python scripts/plot_main_figures.py
```

The standalone script uses only this directory's CSVs, NumPy and Matplotlib.
It does not import the RL implementation or require the original research
checkout. The unified requirements pin Matplotlib3.9.2, matching the backend
version recorded in the archived PDFs. Optional SciencePlots styles are not
activated because the archived figures use default DejaVu Sans styling.

Outputs are `outputs/main_figures/RQ1_aggregated.{pdf,png}` and
`outputs/main_figures/RQ2_aggregated.{pdf,png}`. The accompanying
`reproduction_checks.json` checks all1,440 curve coordinates and all100
shading polygons against the CSVs, with zero numerical discrepancy.

Earlier CSV-only verification also succeeded from a separate directory containing only the
plotting script and two CSVs, in a fresh environment without Torch, AMAGO,
Safety-Gymnasium or pandas. That historical minimal environment is not the
unified installation above, which includes all project dependencies. Fonts,
labels, colors and plotted quantities
match the archived figures. Small renderer/layout differences remain;
the regenerated PDFs are not byte-identical to the originals. This
reproduction preserves the uncertainty limitations described above.
