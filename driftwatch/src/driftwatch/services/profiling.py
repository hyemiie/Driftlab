import numpy as np
import pandas as pd
from pandas.api import types as pdt

from driftwatch.db.models import ColumnDtype

QUANTILES = [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
N_BINS = 10
TOP_N = 20


def _f(x) -> float | None:
    """JSON-safe float: NaN/inf -> None, numpy -> python."""
    if x is None:
        return None
    x = float(x)
    return x if np.isfinite(x) else None


def _rate(missing: int, n_rows: int) -> float:
    return float(missing / n_rows) if n_rows else 0.0


def _numeric_stats(s: pd.Series, n_rows: int) -> dict:
    x = pd.to_numeric(s, errors="coerce").astype("float64")
    x = x.replace([np.inf, -np.inf], np.nan)
    vals = x.dropna().to_numpy()
    count = len(vals)

    stats = {
        "n_rows": n_rows,
        "count": count,
        "missing_rate": _rate(n_rows - count, n_rows),
        "unparseable_count": int(s.notna().sum() - count),
    }
    if count == 0:
        return {**stats, "mean": None, "std": None, "min": None, "max": None,
                "quantiles": {}, "bin_edges": [], "bin_counts": [], "bin_proportions": []}

    # Quantile-based bins. These edges are FIXED here so production data can later
    # be binned identically (PSI needs the same bins on both sides).
    edges = np.unique(np.quantile(vals, np.linspace(0, 1, N_BINS + 1)))
    if len(edges) < 2:  # constant column
        edges = np.array([vals.min() - 0.5, vals.max() + 0.5])
    counts, _ = np.histogram(vals, bins=edges)

    return {
        **stats,
        "mean": _f(vals.mean()),
        "std": _f(vals.std(ddof=1)) if count > 1 else 0.0,
        "min": _f(vals.min()),
        "max": _f(vals.max()),
        "quantiles": {f"{q:g}": _f(v) for q, v in zip(QUANTILES, np.quantile(vals, QUANTILES))},
        "bin_edges": [_f(e) for e in edges],
        "bin_counts": [int(c) for c in counts],
        "bin_proportions": [float(c / count) for c in counts],
    }


def _categorical_stats(s: pd.Series, n_rows: int) -> dict:
    s = s.dropna()
    # 1.0 and 1 should be the same category now and in production
    if pdt.is_float_dtype(s) and len(s) and bool((s % 1 == 0).all()):
        s = s.astype("int64")
    labels = s.astype(str)
    count = len(labels)

    vc = labels.value_counts()
    top = vc.head(TOP_N)
    other = int(vc.iloc[TOP_N:].sum())

    return {
        "n_rows": n_rows,
        "count": count,
        "missing_rate": _rate(n_rows - count, n_rows),
        "n_distinct": int(len(vc)),
        "categories": [
            {"value": str(v), "count": int(c), "proportion": float(c / count)}
            for v, c in top.items()
        ],
        "other_count": other,
        "other_proportion": float(other / count) if count else 0.0,
    }


def _to_datetime(s: pd.Series) -> pd.Series:
    try:
        return pd.to_datetime(s, errors="coerce", utc=True, format="mixed")
    except (TypeError, ValueError):
        return pd.to_datetime(s, errors="coerce", utc=True)


def _datetime_stats(s: pd.Series, n_rows: int) -> dict:
    parsed = _to_datetime(s).dropna()
    count = len(parsed)
    return {
        "n_rows": n_rows,
        "count": count,
        "missing_rate": _rate(n_rows - count, n_rows),
        "unparseable_count": int(s.notna().sum() - count),
        "min": parsed.min().isoformat() if count else None,
        "max": parsed.max().isoformat() if count else None,
    }


def _basic_stats(s: pd.Series, n_rows: int) -> dict:
    """For id/text columns: not drift-tested, but worth recording."""
    count = int(s.notna().sum())
    return {
        "n_rows": n_rows,
        "count": count,
        "missing_rate": _rate(n_rows - count, n_rows),
        "n_distinct": int(s.nunique()),
    }


def compute_reference_profile(df: pd.DataFrame, columns) -> dict[str, dict]:
    """columns: SchemaColumn rows. Returns {column_name: stats_dict}."""
    n_rows = len(df)
    result: dict[str, dict] = {}

    for col in columns:
        if col.dtype == ColumnDtype.ignore:
            continue
        if col.name not in df.columns:
            raise ValueError(f"Column '{col.name}' is not in the uploaded data")

        s = df[col.name]
        if col.dtype == ColumnDtype.numeric:
            stats = _numeric_stats(s, n_rows)
        elif col.dtype == ColumnDtype.categorical:
            stats = _categorical_stats(s, n_rows)
        elif col.dtype == ColumnDtype.datetime:
            stats = _datetime_stats(s, n_rows)
        else:  # id, text
            stats = _basic_stats(s, n_rows)

        result[col.name] = {"kind": col.dtype.value, "role": col.role.value, **stats}

    return result