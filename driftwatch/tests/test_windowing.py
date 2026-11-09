
import pandas as pd
import pytest

from app.windowing import (
    assign_windows,
    window_end_for,
    window_start_for,
)


def test_three_day_batch_produces_three_windows():
    df = pd.DataFrame({
        "timestamp": [
            "2026-01-01 10:00:00",
            "2026-01-02 10:00:00",
            "2026-01-03 10:00:00",
        ]
    })

    result, findings = assign_windows(
        df,
        time_col="timestamp",
        window_size="day",
        uploaded_at="2026-01-04T00:00:00Z",
    )

    assert result["window_start"].nunique() == 3
    assert not findings


def test_midnight_belongs_to_next_day():
    df = pd.DataFrame({
        "timestamp": [
            "2026-01-01 23:59:59",
            "2026-01-02 00:00:00",
        ]
    })

    result, _ = assign_windows(
        df,
        "timestamp",
        "day",
        "2026-01-03T00:00:00Z",
    )

    assert result.iloc[0]["window_start"] == pd.Timestamp(
        "2026-01-01T00:00:00Z"
    )
    assert result.iloc[1]["window_start"] == pd.Timestamp(
        "2026-01-02T00:00:00Z"
    )


def test_week_starts_on_monday():
    timestamps = pd.Series(
        pd.to_datetime([
            "2026-01-07",  # Wednesday
            "2026-01-11",  # Sunday
            "2026-01-12",  # Monday
        ], utc=True)
    )

    result = window_start_for(timestamps, "week")

    assert result.iloc[0] == pd.Timestamp(
        "2026-01-05T00:00:00Z"
    )
    assert result.iloc[1] == pd.Timestamp(
        "2026-01-05T00:00:00Z"
    )
    assert result.iloc[2] == pd.Timestamp(
        "2026-01-12T00:00:00Z"
    )


def test_mixed_timezone_offsets_normalize_to_utc():
    df = pd.DataFrame({
        "timestamp": [
            "2026-01-01T10:00:00+02:00",
            "2026-01-01T08:00:00Z",
            "2026-01-01 08:00:00",
        ]
    })

    result, _ = assign_windows(
        df,
        "timestamp",
        "day",
        "2026-01-03T00:00:00Z",
    )

    expected = pd.Timestamp("2026-01-01T00:00:00Z")

    assert all(result["window_start"] == expected)
    assert str(result["window_start"].dtype) == (
        "datetime64[ns, UTC]"
    )


def test_unparseable_timestamps_are_dropped_and_counted():
    df = pd.DataFrame({
        "timestamp": [
            "2026-01-01T10:00:00Z",
            "invalid-date",
            None,
            "2026-01-02T10:00:00Z",
        ]
    })

    result, findings = assign_windows(
        df,
        "timestamp",
        "day",
        "2026-01-03T00:00:00Z",
    )

    assert len(result) == 2
    assert len(findings) == 1

    finding = findings[0]

    assert finding.code == "unparseable_timestamps"
    assert finding.severity == "warning"
    assert finding.count == 2


def test_batch_window_ignores_data_timestamps():
    df = pd.DataFrame({
        "timestamp": [
            "2026-01-01T10:00:00Z",
            "2026-01-05T10:00:00Z",
            "2026-02-10T10:00:00Z",
        ]
    })

    result, findings = assign_windows(
        df,
        "timestamp",
        "batch",
        "2026-02-11T12:00:00Z",
    )

    assert result["window_start"].nunique() == 1

    assert result.iloc[0]["window_start"] == pd.Timestamp(
        "2026-02-11T12:00:00Z"
    )

    assert not findings


def test_batch_without_time_column():
    df = pd.DataFrame({
        "value": [10, 20, 30]
    })

    result, findings = assign_windows(
        df,
        time_col=None,
        window_size="batch",
        uploaded_at="2026-01-01T12:00:00Z",
    )

    assert len(result) == 3
    assert result["window_start"].nunique() == 1
    assert not findings


def test_window_end_equals_next_window_start():
    first_start = pd.Timestamp("2026-01-01T00:00:00Z")
    second_start = pd.Timestamp("2026-01-02T00:00:00Z")

    first_end = window_end_for(first_start, "day")

    assert first_end == second_start


def test_week_window_has_no_gaps_or_overlaps():
    first_start = pd.Timestamp("2026-01-05T00:00:00Z")
    second_start = pd.Timestamp("2026-01-12T00:00:00Z")

    assert window_end_for(first_start, "week") == second_start


def test_future_timestamps_are_retained_and_warned():
    df = pd.DataFrame({
        "timestamp": [
            "2026-01-01T10:00:00Z",
            "2026-01-05T10:00:00Z",
        ]
    })

    result, findings = assign_windows(
        df,
        "timestamp",
        "day",
        "2026-01-02T00:00:00Z",
    )

    assert len(result) == 2

    future_finding = next(
        f for f in findings
        if f.code == "future_timestamps"
    )

    assert future_finding.count == 1


def test_invalid_window_size_raises_error():
    df = pd.DataFrame({
        "timestamp": ["2026-01-01T10:00:00Z"]
    })

    with pytest.raises(ValueError):
        assign_windows(
            df,
            "timestamp",
            "month",
            "2026-01-02T00:00:00Z",
        )
