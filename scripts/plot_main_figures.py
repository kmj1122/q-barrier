"""Regenerate the two main figures from their shipped plot-ready CSVs only.

No checkpoints, private research checkout, raw trajectories, pandas, or RL
packages are used. Archived means/band widths are not rescaled or smoothed.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ENVIRONMENTS = (
    "DarkRoomCMDP-v0", "SafetyPointGoal1-v0", "SafetyCarGoal1-v0",
    "SafetyHalfCheetahVelocitySample-v0", "SafetyAntVelocitySample-v0",
)
ENVIRONMENT_LABELS = ("SafeDarkRoom", "SafetyPoint", "SafetyCar", "SafetyHalfCheetah", "SafetyAnt")
ALGORITHMS = ("shield", "scared", "cad", "maml", "safemeta")
COLORS = {"shield": "red", "scared": "#00008B", "cad": "green", "maml": "brown", "safemeta": "gray"}
LABELS = {"shield": "Q-Barrier (Ours)", "scared": "SCARED", "cad": "Safe AD",
          "maml": "MAML with penalty", "safemeta": "Safe Meta"}


@dataclass(frozen=True)
class Curve:
    x: np.ndarray
    mean: np.ndarray
    halfwidth: np.ndarray


def load_curves(path: Path, rq: str) -> tuple[dict, int]:
    groups = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"rq", "environment", "algorithm", "metric", "x_value", "mean", "std"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Missing plotting columns in {path.name}")
        rows = list(reader)
    cost_metric = "raw_cost" if rq == "RQ1" else "cost"
    for row in rows:
        key = (row["environment"], row["algorithm"], row["metric"])
        if row["rq"] != rq or key[0] not in ENVIRONMENTS or key[1] not in ALGORITHMS or key[2] not in {"return", cost_metric}:
            raise ValueError(f"Unexpected figure or curve identifier in {path.name}: {key}")
        values = tuple(float(row[name]) for name in ("x_value", "mean", "std"))
        if not all(math.isfinite(v) for v in values) or values[2] < 0:
            raise ValueError(f"Nonfinite value or negative shading width in {path.name}")
        groups.setdefault(key, []).append(values)
    expected = {(env, alg, metric) for env in ENVIRONMENTS for alg in ALGORITHMS
                for metric in ("return", cost_metric)}
    if set(groups) != expected:
        raise ValueError(f"Incomplete set of main-figure curves in {path.name}")
    curves = {}
    for key, values in groups.items():
        values = np.array(sorted(values), dtype=np.float64)
        if np.any(np.diff(values[:, 0]) <= 0):
            raise ValueError(f"Duplicate x coordinates in curve {key}")
        curves[key] = Curve(values[:, 0], values[:, 1], values[:, 2])
    return curves, len(rows)


def apply_paper_style() -> None:
    # The archived PDFs use default DejaVu Sans, not optional SciencePlots serif.
    # Pinning Matplotlib 3.9.2 also reproduces the original layout/font backend.
    matplotlib.rcdefaults()
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "mathtext.fontset": "dejavusans",
        "text.usetex": False, "font.size": 14 * 1.2, "axes.labelsize": 30 * 1.2,
        "xtick.labelsize": 14 * 1.2, "ytick.labelsize": 14 * 1.2,
        "legend.fontsize": 20 * 1.2,
    })


def style_ticks(axis, hide_x: bool) -> None:
    axis.tick_params(axis="both", labelcolor="black")
    if hide_x:
        axis.tick_params(axis="x", bottom=False, labelbottom=False)
    for label in axis.get_xticklabels() + axis.get_yticklabels():
        label.set_fontweight("bold")


def expected_band_vertices(curve: Curve) -> np.ndarray:
    lower = curve.mean - curve.halfwidth
    upper = curve.mean + curve.halfwidth
    return np.concatenate((
        [[curve.x[0], upper[0]]], np.column_stack((curve.x, lower)),
        [[curve.x[-1], upper[-1]]], np.column_stack((curve.x[::-1], upper[::-1])),
        [[curve.x[0], upper[0]]],
    ))


def make_figure(curves: dict, rq: str):
    apply_paper_style()
    fig, axes = plt.subplots(2, 5, figsize=(25, 10))
    cost_metric = "raw_cost" if rq == "RQ1" else "cost"
    artists = {}
    for row, metric in enumerate(("return", cost_metric)):
        for column, env in enumerate(ENVIRONMENTS):
            axis = axes[row, column]
            for alg in ALGORITHMS:
                curve = curves[(env, alg, metric)]
                line, = axis.plot(curve.x, curve.mean, label=LABELS[alg], linewidth=2, color=COLORS[alg])
                band = axis.fill_between(curve.x, curve.mean - curve.halfwidth,
                                         curve.mean + curve.halfwidth, alpha=.2, color=COLORS[alg])
                artists[(env, alg, metric)] = (line, band)
            if rq == "RQ1":
                limit = 50 if column == 0 else 15 if column in (1, 2) else 10
                ticks = [0, 10, 20, 30, 40, 50] if limit == 50 else [0, 5, 10, 15] if limit == 15 else [0, 5, 10]
                axis.set_xticks(ticks)
                axis.set_xlim(0, limit)
            else:
                if column == 0:
                    limits, ticks = (1., 15.), [1., 4.5, 8., 11.5, 15.]
                elif column in (1, 2):
                    limits, ticks = (10., 50.), [10, 20, 30, 40, 50]
                else:
                    limits, ticks = (0., 5.), [0, 1.25, 2.5, 3.75, 5]
                # Retain even clipped CSV points: they contribute to y-autoscale.
                axis.set_xlim(*limits)
                axis.set_xticks(ticks)
            style_ticks(axis, hide_x=row == 0)
            axis.grid(True, alpha=.3)
            if row == 0:
                axis.set_title(ENVIRONMENT_LABELS[column], fontsize=22 * 1.2)
            else:
                axis.set_xlabel(r"Episode Index $k$" if rq == "RQ1" else r"Cost Budget $\delta$",
                                fontsize=20 * 1.4)
            if column == 0:
                label = ("Return" if row == 0 else "Cost") if rq == "RQ1" else (
                    r"$\sum_{k=1}^{K} G(\tau_k)$" if row == 0
                    else r"$\frac{1}{K}\sum_{k=1}^{K} G_c(\tau_k)$")
                axis.set_ylabel(label, fontsize=20 * 1.4)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, .05),
               ncol=len(ALGORITHMS), fontsize=24 * 1.2)
    fig.tight_layout(rect=[0, .05, 1, 1])
    return fig, axes, artists


def verify_artists(curves: dict, artists: dict) -> dict:
    """Check actual Matplotlib line and polygon arrays against every CSV value."""
    line_error = band_error = 0.
    for key, curve in curves.items():
        line, band = artists[key]
        expected_line = np.column_stack((curve.x, curve.mean))
        actual_line = np.column_stack((line.get_xdata(), line.get_ydata()))
        np.testing.assert_array_equal(actual_line, expected_line)
        expected_band = expected_band_vertices(curve)
        actual_band = band.get_paths()[0].vertices
        np.testing.assert_array_equal(actual_band, expected_band)
        line_error = max(line_error, float(np.max(np.abs(actual_line - expected_line))))
        band_error = max(band_error, float(np.max(np.abs(actual_band - expected_band))))
    return {"curves_checked": len(artists), "bands_checked": len(artists),
            "maximum_line_coordinate_error": line_error, "maximum_band_coordinate_error": band_error}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/figures")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/main_figures")
    parser.add_argument("--figure", choices=("RQ1", "RQ2", "both"), default="both")
    parser.add_argument("--pdf-only", action="store_true", help="Skip PNG previews")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {"matplotlib_version": matplotlib.__version__, "rescaling": False,
              "smoothing": False, "raw_experiments_loaded": False,
              "uncertainty": "Archived shading widths, not newly validated SEM"}
    for rq in (("RQ1", "RQ2") if args.figure == "both" else (args.figure,)):
        path = args.data_dir / f"{rq}_aggregated_plot_values.csv"
        curves, row_count = load_curves(path, rq)
        fig, axes, artists = make_figure(curves, rq)
        verified = verify_artists(curves, artists)
        output = args.output_dir / f"{rq}_aggregated.pdf"
        fig.savefig(output, dpi=300, bbox_inches="tight")
        if not args.pdf_only:
            fig.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
        report[rq] = {"input_csv": path.name, "csv_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                      "rows_checked": row_count, **verified, "output_pdf": output.name,
                      "output_pdf_sha256": hashlib.sha256(output.read_bytes()).hexdigest()}
        plt.close(fig)
        print(f"{rq}: verified {row_count} points, {len(curves)} curves and bands; wrote {output}")
    (args.output_dir / "reproduction_checks.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
