"""Per-window profiling against reference bin edges.

Turns the rows of one window into a small, ADDITIVE summary:

  numeric      counts per reference bin (+ underflow/overflow), missing/invalid
               counts, and sum / sum-of-squares / min / max (for mean and std)
  categorical  counts per reference category, unseen categories (capped),
               missing count

Additive means summary(A + B) == merge(summary(A), summary(B)). That is what
lets /predict-log fold each incoming batch into a window with one upsert
instead of storing raw rows.

Counts array layout for numeric columns, with k reference bins:
    [underflow, bin_0, ..., bin_{k-1}, overflow]      length k + 2
Bins are [e_i, e_{i+1}); the last bin also includes its right edge.
Summaries from different references must never be merged: bin counts would
line up by position but mean different ranges.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Mapping, Union

import numpy as np
import pandas as pd

MAX_UNSEEN = 20  # distinct unseen categories tracked per window per column


# ---------- Reference (what a window is compared against) ----------


@dataclass(frozen=True)
class NumericRef:
    edges: tuple[float, ...]  # ascending, len >= 2 -> len - 1 bins


@dataclass(frozen=True)
class CategoricalRef:
    categories: tuple[str, ...]


ColumnRef = Union[NumericRef, CategoricalRef]


def _is_categorical(s: pd.Series) -> bool:
    if pd.api.types.is_bool_dtype(s):
        return True
    non_null = s.dropna()
    if pd.api.types.is_numeric_dtype(s):
        integral = len(non_null) > 0 and bool((non_null == non_null.round()).all())
        return integral and non_null.nunique() <= 10
    return True


def _labels(s: pd.Series) -> pd.Series:
    """String labels; integer-coded values become '1', not '1.0'."""
    if pd.api.types.is_bool_dtype(s):
        return s.astype("string")
    if pd.api.types.is_numeric_dtype(s):
        non_null = s.dropna()
        if len(non_null) and bool((non_null == non_null.round()).all()):
            return s.astype("Int64").astype("string")
    return s.astype("string")


def build_reference(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    max_bins: int = 10,
    max_categories: int = 20,
) -> dict[str, ColumnRef]:
    """Quantile bins for numeric columns, top-N categories for categorical ones.

    Skip this if your reference profile already stores bin edges; just wrap
    them in NumericRef / CategoricalRef. Datetime and free-text columns are
    skipped.
    """
    ref: dict[str, ColumnRef] = {}
    for col in columns or list(df.columns):
        s = df[col]
        if pd.api.types.is_datetime64_any_dtype(s):
            continue
        if _is_categorical(s):
            labels = _labels(s).dropna()
            if labels.nunique() > 50:
                continue  # free text, not a category
            top = labels.value_counts().index[:max_categories]
            ref[col] = CategoricalRef(tuple(sorted(str(c) for c in top)))
        else:
            x = pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)
            x = x[np.isfinite(x)]
            if len(x) == 0:
                continue
            edges = np.unique(np.quantile(x, np.linspace(0, 1, max_bins + 1)))
            if len(edges) < 2:  # constant column
                edges = np.array([edges[0], np.nextafter(edges[0], np.inf)])
            ref[col] = NumericRef(tuple(float(e) for e in edges))
    return ref


# ---------- Summaries ----------


@dataclass
class NumericSummary:
    counts: list[int]
    missing: int = 0   # null in the data
    invalid: int = 0   # present but not parseable as a number
    n_finite: int = 0
    total: float = 0.0
    total_sq: float = 0.0
    minimum: float | None = None
    maximum: float | None = None
    absent: bool = False  # the column was not in the data at all
    kind: str = "numeric"

    @property
    def n(self) -> int:
        return int(sum(self.counts))

    @property
    def mean(self) -> float | None:
        return self.total / self.n_finite if self.n_finite else None

    @property
    def std(self) -> float | None:
        if not self.n_finite:
            return None
        var = self.total_sq / self.n_finite - (self.total / self.n_finite) ** 2
        return math.sqrt(max(var, 0.0))


@dataclass
class CategoricalSummary:
    counts: dict[str, int]
    unseen: dict[str, int] = field(default_factory=dict)  # top MAX_UNSEEN by count
    unseen_other: int = 0                                  # the rest, lumped
    missing: int = 0
    absent: bool = False
    kind: str = "categorical"

    @property
    def unseen_total(self) -> int:
        return sum(self.unseen.values()) + self.unseen_other

    @property
    def n(self) -> int:
        return sum(self.counts.values()) + self.unseen_total


ColumnSummary = Union[NumericSummary, CategoricalSummary]


@dataclass
class WindowSummary:
    n_rows: int
    columns: dict[str, ColumnSummary]


# ---------- Profiling ----------


def _profile_numeric(s: pd.Series | None, ref: NumericRef, n_rows: int) -> NumericSummary:
    k = len(ref.edges) - 1
    if s is None:
        return NumericSummary(counts=[0] * (k + 2), missing=n_rows, absent=True)

    raw_missing = s.isna()
    num = pd.to_numeric(s, errors="coerce")
    invalid = int((num.isna() & ~raw_missing).sum())
    x = num.dropna().to_numpy(dtype=float)

    edges = np.asarray(ref.edges, dtype=float)
    idx = np.searchsorted(edges, x, side="right")  # 0 = underflow ... k+1 = overflow
    idx[x == edges[-1]] = k                        # right edge belongs to the last bin
    counts = np.bincount(idx, minlength=k + 2).astype(int).tolist()

    finite = x[np.isfinite(x)]
    return NumericSummary(
        counts=counts,
        missing=int(raw_missing.sum()),
        invalid=invalid,
        n_finite=int(len(finite)),
        total=float(finite.sum()),
        total_sq=float((finite**2).sum()),
        minimum=float(finite.min()) if len(finite) else None,
        maximum=float(finite.max()) if len(finite) else None,
    )


def _cap_unseen(items: dict[str, int], extra_other: int = 0) -> tuple[dict[str, int], int]:
    ranked = sorted(items.items(), key=lambda kv: (-kv[1], kv[0]))
    keep = dict(ranked[:MAX_UNSEEN])
    other = extra_other + sum(c for _, c in ranked[MAX_UNSEEN:])
    return keep, other


def _profile_categorical(s: pd.Series | None, ref: CategoricalRef, n_rows: int) -> CategoricalSummary:
    if s is None:
        return CategoricalSummary(counts={c: 0 for c in ref.categories}, missing=n_rows, absent=True)

    labels = _labels(s)
    vc = labels.dropna().value_counts()
    counts_by_label = {str(k): int(v) for k, v in vc.items()}
    known = set(ref.categories)
    unseen, other = _cap_unseen({k: v for k, v in counts_by_label.items() if k not in known})
    return CategoricalSummary(
        counts={c: counts_by_label.get(c, 0) for c in ref.categories},
        unseen=unseen,
        unseen_other=other,
        missing=int(labels.isna().sum()),
    )


def profile_window(df: pd.DataFrame, ref: Mapping[str, ColumnRef]) -> WindowSummary:
    """Summarise one window's rows against the reference."""
    n_rows = len(df)
    cols: dict[str, ColumnSummary] = {}
    for name, r in ref.items():
        s = df[name] if name in df.columns else None
        if isinstance(r, NumericRef):
            cols[name] = _profile_numeric(s, r, n_rows)
        else:
            cols[name] = _profile_categorical(s, r, n_rows)
    return WindowSummary(n_rows=n_rows, columns=cols)


# ---------- Merging (incremental updates) ----------


def _opt(fn, a, b):
    vals = [v for v in (a, b) if v is not None]
    return fn(vals) if vals else None


def merge_columns(a: ColumnSummary, b: ColumnSummary) -> ColumnSummary:
    if isinstance(a, NumericSummary) and isinstance(b, NumericSummary):
        if len(a.counts) != len(b.counts):
            raise ValueError("Cannot merge numeric summaries built from different bin edges.")
        return NumericSummary(
            counts=[x + y for x, y in zip(a.counts, b.counts)],
            missing=a.missing + b.missing,
            invalid=a.invalid + b.invalid,
            n_finite=a.n_finite + b.n_finite,
            total=a.total + b.total,
            total_sq=a.total_sq + b.total_sq,
            minimum=_opt(min, a.minimum, b.minimum),
            maximum=_opt(max, a.maximum, b.maximum),
            absent=a.absent and b.absent,
        )
    if isinstance(a, CategoricalSummary) and isinstance(b, CategoricalSummary):
        if set(a.counts) != set(b.counts):
            raise ValueError("Cannot merge categorical summaries built from different categories.")
        unseen = dict(a.unseen)
        for k, v in b.unseen.items():
            unseen[k] = unseen.get(k, 0) + v
        unseen, other = _cap_unseen(unseen, a.unseen_other + b.unseen_other)
        return CategoricalSummary(
            counts={c: a.counts[c] + b.counts[c] for c in a.counts},
            unseen=unseen,
            unseen_other=other,
            missing=a.missing + b.missing,
            absent=a.absent and b.absent,
        )
    raise ValueError("Cannot merge numeric and categorical summaries.")


def merge_windows(a: WindowSummary, b: WindowSummary) -> WindowSummary:
    names = list(dict.fromkeys([*a.columns, *b.columns]))
    cols = {
        n: merge_columns(a.columns[n], b.columns[n]) if n in a.columns and n in b.columns
        else (a.columns.get(n) or b.columns[n])
        for n in names
    }
    return WindowSummary(n_rows=a.n_rows + b.n_rows, columns=cols)


# ---------- Serialisation (store as JSON/JSONB) ----------


def summary_to_dict(s: ColumnSummary) -> dict:
    return asdict(s)


def summary_from_dict(d: dict) -> ColumnSummary:
    d = dict(d)
    kind = d.pop("kind")
    return NumericSummary(**d) if kind == "numeric" else CategoricalSummary(**d)