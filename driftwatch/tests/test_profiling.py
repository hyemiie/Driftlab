import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from driftwatch.db.models import ColumnDtype as D
from driftwatch.db.models import ColumnRole as R
from driftwatch.services.profiling import TOP_N, compute_reference_profile

rng = np.random.default_rng(1)


def run(df, **spec):
    """spec: column name -> dtype, or (dtype, role)."""
    cols = []
    for name, v in spec.items():
        dtype, role = v if isinstance(v, tuple) else (v, R.feature)
        cols.append(SimpleNamespace(name=name, dtype=dtype, role=role))
    return compute_reference_profile(df, cols)


# ---- numeric ----
def test_numeric_summary_matches_numpy():
    x = rng.normal(50, 10, 500)
    s = run(pd.DataFrame({"x": x}), x=D.numeric)["x"]
    assert s["count"] == 500 and s["missing_rate"] == 0.0
    assert s["mean"] == pytest.approx(x.mean())
    assert s["std"] == pytest.approx(x.std(ddof=1))
    assert s["min"] == pytest.approx(x.min()) and s["max"] == pytest.approx(x.max())
    assert s["quantiles"]["0.5"] == pytest.approx(np.quantile(x, 0.5))
    assert s["quantiles"]["0.99"] == pytest.approx(np.quantile(x, 0.99))


def test_bins_are_consistent_and_roughly_equal():
    x = rng.normal(50, 10, 500)
    s = run(pd.DataFrame({"x": x}), x=D.numeric)["x"]
    edges, counts = s["bin_edges"], s["bin_counts"]
    assert len(edges) == len(counts) + 1
    assert all(b > a for a, b in zip(edges, edges[1:]))  # strictly increasing
    assert edges[0] == pytest.approx(x.min()) and edges[-1] == pytest.approx(x.max())
    assert sum(counts) == 500
    assert all(45 <= c <= 55 for c in counts)  # quantile bins ~ equal mass
    assert sum(s["bin_proportions"]) == pytest.approx(1.0)


def test_stored_bins_can_be_reused_on_new_data():
    x = rng.normal(50, 10, 500)
    s = run(pd.DataFrame({"x": x}), x=D.numeric)["x"]
    counts, _ = np.histogram(rng.normal(50, 10, 300), bins=s["bin_edges"])
    assert len(counts) == len(s["bin_counts"])


def test_missing_values_are_counted():
    x = rng.normal(size=500)
    x[:100] = np.nan
    s = run(pd.DataFrame({"x": x}), x=D.numeric)["x"]
    assert s["count"] == 400 and s["missing_rate"] == pytest.approx(0.2)


def test_infinities_are_treated_as_missing():
    x = np.array([1.0, 2.0, np.inf, -np.inf, np.nan, 3.0] * 10)
    s = run(pd.DataFrame({"x": x}), x=D.numeric)["x"]
    assert s["count"] == 30
    assert s["max"] == 3.0 and s["min"] == 1.0


def test_unparseable_numeric_values_are_reported():
    s = run(pd.DataFrame({"x": ["1", "2", "x", "4"] * 10}), x=D.numeric)["x"]
    assert s["count"] == 30 and s["unparseable_count"] == 10


def test_constant_numeric_column():
    s = run(pd.DataFrame({"x": [5.0] * 40}), x=D.numeric)["x"]
    assert s["std"] == 0.0
    assert s["bin_counts"] == [40] and len(s["bin_edges"]) == 2


def test_all_missing_numeric_column():
    s = run(pd.DataFrame({"x": [np.nan] * 40}), x=D.numeric)["x"]
    assert s["count"] == 0 and s["mean"] is None
    assert s["bin_edges"] == [] and s["bin_counts"] == []


# ---- categorical ----
def test_categories_cap_at_top_n_with_other_bucket():
    values = [f"c{i}" for i in range(30) for _ in range(10)]
    s = run(pd.DataFrame({"x": values}), x=D.categorical)["x"]
    assert s["n_distinct"] == 30
    assert len(s["categories"]) == TOP_N
    assert s["other_count"] == (30 - TOP_N) * 10
    total = sum(c["proportion"] for c in s["categories"]) + s["other_proportion"]
    assert total == pytest.approx(1.0)


def test_categorical_missing_rate():
    values = ["a", "b", None, "a"] * 25
    s = run(pd.DataFrame({"x": values}), x=D.categorical)["x"]
    assert s["missing_rate"] == pytest.approx(0.25)
    assert s["count"] == 75


def test_whole_number_floats_become_clean_labels():
    s = run(pd.DataFrame({"x": [1.0, 2.0, np.nan, 1.0] * 10}), x=D.categorical)["x"]
    assert {c["value"] for c in s["categories"]} == {"1", "2"}


def test_boolean_labels():
    s = run(pd.DataFrame({"x": [True, False, True] * 10}), x=D.categorical)["x"]
    assert {c["value"] for c in s["categories"]} == {"True", "False"}


# ---- datetime ----
def test_datetime_stats():
    values = ["2024-01-01", "2024-03-15", "not a date", None] * 10
    s = run(pd.DataFrame({"t": values}), t=D.datetime)["t"]
    assert s["count"] == 20 and s["unparseable_count"] == 10
    assert s["min"].startswith("2024-01-01") and s["max"].startswith("2024-03-15")


# ---- structure ----
def test_ignored_columns_are_skipped():
    df = pd.DataFrame({"a": rng.normal(size=40), "b": rng.normal(size=40)})
    assert list(run(df, a=D.numeric, b=D.ignore)) == ["a"]


def test_id_and_text_columns_get_basic_stats_only():
    df = pd.DataFrame({"id": range(40), "note": ["hello"] * 40})
    out = run(df, id=D.id, note=D.text)
    for name in ("id", "note"):
        assert "bin_edges" not in out[name] and "quantiles" not in out[name]
        assert out[name]["n_distinct"] >= 1


def test_kind_and_role_are_recorded():
    df = pd.DataFrame({"y": rng.integers(0, 2, 40)})
    s = run(df, y=(D.categorical, R.target))["y"]
    assert s["kind"] == "categorical" and s["role"] == "target"
    assert s["n_rows"] == 40


def test_unknown_column_raises():
    with pytest.raises(ValueError, match="not in"):
        run(pd.DataFrame({"a": [1, 2, 3]}), ghost=D.numeric)


def test_output_is_json_safe_for_postgres():
    """JSONB rejects NaN and numpy types, so a messy frame must serialize strictly."""
    df = pd.DataFrame(
        {
            "num": [1.0, np.nan, np.inf, 4.0] * 10,
            "empty": [np.nan] * 40,
            "cat": ["a", None, "b", "a"] * 10,
            "when": ["2024-01-01", "garbage", None, "2024-02-01"] * 10,
            "id": range(40),
        }
    )
    out = run(
        df, num=D.numeric, empty=D.numeric, cat=D.categorical, when=D.datetime, id=D.id
    )
    json.dumps(out, allow_nan=False)  # raises on NaN/inf or non-serializable types