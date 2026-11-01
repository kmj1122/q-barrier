"""Regression checks for CSV-only main-figure rendering."""
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

from scripts.plot_main_figures import (
    Curve, expected_band_vertices, load_curves, make_figure, verify_artists,
)


DATA = Path(__file__).resolve().parents[1] / "data/figures"


@pytest.mark.parametrize("rq", ("RQ1", "RQ2"))
def test_all_exported_curve_and_band_coordinates_are_preserved(rq):
    curves, count = load_curves(DATA / f"{rq}_aggregated_plot_values.csv", rq)
    figure, axes, artists = make_figure(curves, rq)
    try:
        result = verify_artists(curves, artists)
        assert count == (1000 if rq == "RQ1" else 440)
        assert result["curves_checked"] == result["bands_checked"] == 50
        assert result["maximum_line_coordinate_error"] == result["maximum_band_coordinate_error"] == 0
        assert len(axes.flat) == 10
        assert tuple(axes[0, 0].get_xlim()) == ((0, 50) if rq == "RQ1" else (1, 15))
    finally:
        plt.close(figure)


def test_band_geometry_against_explicit_fixture():
    curve = Curve(np.array([0., 1.]), np.array([2., 3.]), np.array([.5, .25]))
    expected = np.array([[0, 2.5], [0, 1.5], [1, 2.75], [1, 3.25],
                         [1, 3.25], [0, 2.5], [0, 2.5]])
    np.testing.assert_array_equal(expected_band_vertices(curve), expected)


def test_validation_catches_a_second_scaling_of_a_curve():
    curve = Curve(np.array([0., 1.]), np.array([2., 3.]), np.array([.5, .25]))
    figure, axis = plt.subplots()
    line, = axis.plot(curve.x, curve.mean * .5)
    band = axis.fill_between(curve.x, curve.mean - curve.halfwidth, curve.mean + curve.halfwidth)
    try:
        with pytest.raises(AssertionError):
            verify_artists({"fixture": curve}, {"fixture": (line, band)})
    finally:
        plt.close(figure)


@pytest.mark.parametrize("change", ("missing_curve", "duplicate_point", "nonfinite", "negative_band"))
def test_bad_plot_exports_are_rejected(tmp_path, change):
    with (DATA / "RQ1_aggregated_plot_values.csv").open() as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames
        rows = list(reader)
    if change == "missing_curve":
        rows = [r for r in rows if not (r["environment"] == "DarkRoomCMDP-v0"
                                       and r["algorithm"] == "shield" and r["metric"] == "return")]
    elif change == "duplicate_point":
        rows.append(dict(rows[0]))
    elif change == "nonfinite":
        rows[0]["mean"] = "nan"
    else:
        rows[0]["std"] = "-0.1"
    path = tmp_path / "invalid.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError):
        load_curves(path, "RQ1")
