import pandas as pd


def to_datetime(s: pd.Series) -> pd.Series:
    """Parse to timezone-aware UTC timestamps. Unparseable values become NaT.

    Shared by profiling, validation and windowing, so all three agree on what
    counts as a valid date.
    """
    try:
        return pd.to_datetime(s, errors="coerce", utc=True, format="mixed")
    except (TypeError, ValueError):  # pandas < 2 has no format="mixed"
        return pd.to_datetime(s, errors="coerce", utc=True)