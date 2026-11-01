"""Check archived plotting exports without treating them as new experiments."""
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import pytest


DATA = Path(__file__).resolve().parents[1] / "data/figures"


@pytest.mark.parametrize("rq,n_rows", [("RQ1", 1000), ("RQ2", 440)])
def test_plotting_export_schema_values_and_hash(rq, n_rows):
    path = DATA / f"{rq}_aggregated_plot_values.csv"
    provenance = json.loads((DATA / "provenance.json").read_text())
    assert hashlib.sha256(path.read_bytes()).hexdigest() == provenance[rq]["csv_sha256"]
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    assert len(rows) == n_rows
    assert {r["rq"] for r in rows} == {rq}
    assert len({r["environment"] for r in rows}) == 5
    assert {r["algorithm"] for r in rows} == {"shield", "scared", "cad", "maml", "safemeta"}
    keys = [(r["environment"], r["algorithm"], r["metric"], r["x_value"]) for r in rows]
    assert len(set(keys)) == len(rows)
    groups = defaultdict(list)
    for row in rows:
        assert all(math.isfinite(float(row[k])) for k in ("x_value", "mean", "std"))
        assert float(row["std"]) >= 0
        groups[(row["environment"], row["algorithm"], row["metric"])].append(float(row["x_value"]))
    assert len(groups) == 50
    assert all(all(x < y for x, y in zip(v, v[1:])) for v in groups.values())
    if rq == "RQ1":
        assert all(values == list(range(len(values))) for values in groups.values())
    else:
        assert sorted(map(len, groups.values())) == [4, 4] + [9] * 48
