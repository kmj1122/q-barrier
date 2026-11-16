# Main-figure data

These files hold the plotted values of the two main figures in the paper.

| File | Figure | Rows |
|---|---|---:|
| `RQ1_aggregated_plot_values.csv` | Return and cost over in-context episodes | 1,000 |
| `RQ2_aggregated_plot_values.csv` | Return and cost over cost budgets | 440 |

Each file covers five environments and five methods: Q-Barrier, SCARED, Safe AD, MAML with penalty and SafeMeta.

## Columns

- `rq`: RQ1 or RQ2.
- `environment`, `environment_display`: environment key and panel label.
- `algorithm`, `algorithm_display`: method key and legend label.
- `metric`, `metric_label`: return or cost, and the axis label.
- `x_name`, `x_value`: the x-axis variable and its value. RQ1 episode indices start at 0.
- `mean`: the plotted value.
- `std`: the half-width of the shaded band, drawn as `mean ± std`.
- `source_mean_column`, `source_std_column`: column names in the original result tables.
- `notes`: scale factors used when plotting.

Scaling and smoothing are already applied to `mean` and `std`. Do not apply them again.

## Redraw the figures

From the repository root:

```bash
python scripts/plot_main_figures.py
```

This needs only NumPy and Matplotlib. The figures are written to `outputs/main_figures/` as PDF and PNG.
