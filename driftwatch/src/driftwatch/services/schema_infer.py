import re
import warnings
from dataclasses import dataclass

import pandas as pd
from pandas.api import types as pdt

from driftwatch.db.models import ColumnDtype

CATEGORICAL_MAX_DISTINCT = 15   # integer columns with <= this many values -> categorical
ID_UNIQUE_RATIO = 0.95          # "nearly every value is unique"
DATE_PARSE_RATIO = 0.95         # share of sampled strings that must parse as dates
LONG_TEXT_LEN = 50              # average length above this -> free text

_ID_NAME = re.compile(r"(?:^|[_\s-])(id|uuid|guid|key|index)(?:$|[_\s-])")


@dataclass
class InferredColumn:
    name: str
    dtype: ColumnDtype
    reason: str


def _id_like_name(name: str) -> bool:
    return bool(_ID_NAME.search(name.lower()))


def _looks_like_dates(s: pd.Series) -> bool:
    sample = s.astype(str)
    if len(sample) > 200:
        sample = sample.sample(200, random_state=0)
    # cheap guard so strings like "12" or "abc" never reach the date parser
    if sample.str.contains(r"[-/:]").mean() < DATE_PARSE_RATIO:
        return False
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
        except (TypeError, ValueError):  # older pandas without format="mixed"
            parsed = pd.to_datetime(sample, errors="coerce")
    return bool(parsed.notna().mean() >= DATE_PARSE_RATIO)


def _infer_column(name: str, s: pd.Series) -> InferredColumn:
    def out(dtype: ColumnDtype, reason: str) -> InferredColumn:
        return InferredColumn(name, dtype, reason)

    nn = s.dropna()
    n = len(nn)
    if n == 0:
        return out(ColumnDtype.ignore, "all values are missing")

    nunique = nn.nunique()
    if nunique <= 1:
        return out(ColumnDtype.ignore, "constant column (single distinct value)")

    if pdt.is_bool_dtype(s):
        return out(ColumnDtype.categorical, "boolean column")

    if pdt.is_datetime64_any_dtype(s):
        return out(ColumnDtype.datetime, "datetime column")

    ratio = nunique / n

    if pdt.is_numeric_dtype(s):
        integer_valued = bool((nn % 1 == 0).all())
        if (
            integer_valued
            and ratio >= ID_UNIQUE_RATIO
            and (_id_like_name(name) or nn.is_monotonic_increasing)
        ):
            return out(ColumnDtype.id, "unique integers that look like an identifier")
        if integer_valued and nunique <= CATEGORICAL_MAX_DISTINCT:
            return out(
                ColumnDtype.categorical,
                f"only {nunique} distinct integer values, treated as categorical",
            )
        return out(ColumnDtype.numeric, f"numeric with {nunique} distinct values")

    # everything else is string-like
    if _looks_like_dates(nn):
        return out(ColumnDtype.datetime, "values parse as dates")

    strs = nn.astype(str)
    avg_len = strs.str.len().mean()

    if ratio >= ID_UNIQUE_RATIO:
        has_spaces = strs.str.contains(r"\s").any()
        if _id_like_name(name) or (avg_len <= 40 and not has_spaces):
            return out(ColumnDtype.id, "nearly all values unique, short, no spaces")
        return out(ColumnDtype.text, "nearly all values unique free text")

    if avg_len > LONG_TEXT_LEN:
        return out(ColumnDtype.text, f"long strings (average {avg_len:.0f} characters)")

    return out(ColumnDtype.categorical, f"text with {nunique} distinct values")


def infer_schema(df: pd.DataFrame) -> list[InferredColumn]:
    return [_infer_column(c, df[c]) for c in df.columns]