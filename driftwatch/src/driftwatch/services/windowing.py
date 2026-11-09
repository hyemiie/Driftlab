
from dataclasses import dataclass
from datetime import datetime

import pandas as pd

from .utils import to_datetime


# ---------- Findings ----------


@dataclass
class Finding:
    code: str
    message: str
    severity: str
    count: int = 0


# ---------- Constants ----------

VALID_WINDOW_SIZES = {"day", "week", "batch"}


# ---------- Timestamp helpers ----------


def _utc_timestamp(value) -> pd.Timestamp:
    """Convert a scalar timestamp to UTC."""
    result = to_datetime(pd.Series([value]), utc=True)

    if result.isna().iloc[0]:
        raise ValueError("Timestamp could not be parsed.")

    return pd.Timestamp(result.iloc[0])


def window_start_for(
    ts: pd.Series,
    window_size: str,
) -> pd.Series:
    """
    Return UTC window start timestamps.

    Weekly windows start on Monday at 00:00 UTC.
    """
    if window_size not in {"day", "week"}:
        raise ValueError(
            "window_size must be 'day' or 'week'."
        )

    # Ensure all timestamps are UTC-aware.
    ts = to_datetime(ts, utc=True)

    if window_size == "day":
        return ts.dt.floor("D")

    # Weeks start on Monday at 00:00 UTC.
    # This convention must remain consistent across all charts
    # and window-based aggregations.
    day_start = ts.dt.floor("D")
    return day_start - pd.to_timedelta(
        day_start.dt.weekday,
        unit="D",
    )


def window_end_for(
    start: pd.Timestamp,
    window_size: str,
) -> pd.Timestamp:
    """
    Return the exclusive end of a window.

    A row exactly at window_end belongs to the next window.
    """
    start = _utc_timestamp(start)

    if window_size == "day":
        return start + pd.Timedelta(days=1)

    if window_size == "week":
        return start + pd.Timedelta(days=7)

    if window_size == "batch":
        # A batch is an individual upload, not a time interval.
        # Use a minimal positive interval for database storage.
        return start + pd.Timedelta(microseconds=1)

    raise ValueError(
        "window_size must be 'day', 'week', or 'batch'."
    )


# ---------- Window assignment ----------


def assign_windows(
    df: pd.DataFrame,
    time_col: str | None,
    window_size: str,
    uploaded_at,
) -> tuple[pd.DataFrame, list[Finding]]:
    """
    Assign each row to a UTC window.

    Returns:
        A copy of the DataFrame with a window_start column.
        A list of findings for rows that could not be placed
        or timestamps that appear to be in the future.
    """
    if window_size not in VALID_WINDOW_SIZES:
        raise ValueError(
            "window_size must be 'day', 'week', or 'batch'."
        )

    if time_col is None and window_size != "batch":
        raise ValueError(
            "Without a time column, window_size must be 'batch'."
        )

    if time_col is not None and time_col not in df.columns:
        raise ValueError(
            f"Time column '{time_col}' does not exist."
        )

    result = df.copy()
    findings: list[Finding] = []

    upload_ts = _utc_timestamp(uploaded_at)

    # Batch mode: all rows from this upload share one window.
    if window_size == "batch":
        result["window_start"] = pd.Series(
            upload_ts,
            index=result.index,
            dtype="datetime64[ns, UTC]",
        )
        return result, findings

    # A time column is required for day/week windows.
    if time_col is None:
        raise ValueError(
            "A time column is required for day/week windows."
        )

    # Parse all timestamps using the shared helper.
    # utc=True normalizes naive and offset-aware values to UTC.
    parsed = to_datetime(
        result[time_col],
        utc=True,
    )

    invalid_mask = parsed.isna()
    invalid_count = int(invalid_mask.sum())

    if invalid_count:
        findings.append(
            Finding(
                code="unparseable_timestamps",
                message=(
                    f"Dropped {invalid_count} rows because "
                    "their timestamps could not be parsed."
                ),
                severity="warning",
                count=invalid_count,
            )
        )

        # Do not silently discard rows.
        result = result.loc[~invalid_mask].copy()
        parsed = parsed.loc[~invalid_mask]

    # Future timestamps are retained but reported.
    future_mask = parsed > upload_ts
    future_count = int(future_mask.sum())

    if future_count:
        findings.append(
            Finding(
                code="future_timestamps",
                message=(
                    f"Found {future_count} rows with timestamps "
                    "later than the upload timestamp. "
                    "These rows were retained."
                ),
                severity="warning",
                count=future_count,
            )
        )

    # Assign UTC window starts.
    result["window_start"] = window_start_for(
        parsed,
        window_size,
    )

    return result, findings
