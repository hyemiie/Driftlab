
from dataclasses import dataclass

import pandas as pd
import pytest

from driftwatch.services.validation import validate_against_schema
from driftwatch.db.models import ColumnDtype, ColumnRole

@dataclass
class SchemaColumn:
    name: str
    dtype: ColumnDtype
    role: ColumnRole

@pytest.fixture
def schema():
    return [
        SchemaColumn("Age", ColumnDtype.numeric, ColumnRole.feature),
        SchemaColumn("created_at", ColumnDtype.datetime, ColumnRole.time),
        SchemaColumn("customer_id", ColumnDtype.id, ColumnRole.id),
        SchemaColumn("prediction", ColumnDtype.numeric, ColumnRole.prediction),
        SchemaColumn("target", ColumnDtype.numeric, ColumnRole.target),
        SchemaColumn("internal", ColumnDtype.ignore, ColumnRole.feature),
    ]


@pytest.fixture
def reference_stats():
    return {
        "Age": {"missing_rate": 0.0},
        "created_at": {"missing_rate": 0.0},
        "customer_id": {"missing_rate": 0.0},
        "prediction": {"missing_rate": 0.0},
        "target": {"missing_rate": 0.0},
    }


@pytest.fixture
def clean_df():
    return pd.DataFrame({
        "Age": [20, 30, 40],
        "created_at": [
            "2025-01-01",
            "2025-01-02",
            "2025-01-03",
        ],
        "customer_id": [1, 2, 3],
        "prediction": [0.1, 0.2, 0.3],
        "internal": ["a", "b", "c"],
    })


def test_clean_batch_has_no_findings(
    schema, reference_stats, clean_df
):
    findings = validate_against_schema(
        clean_df, schema, reference_stats
    )

    assert findings == []


def test_missing_feature_is_error_but_missing_target_is_ignored(
    schema, reference_stats, clean_df
):
    df = clean_df.drop(columns=["Age"])

    findings = validate_against_schema(
        df, schema, reference_stats
    )

    missing = [
        f for f in findings
        if f.code == "missing_column"
    ]

    assert len(missing) == 1
    assert missing[0].column == "Age"
    assert missing[0].severity == "error"

    # Target is absent but should not produce a finding.
    assert not any(
        f.column == "target" for f in findings
    )


def test_missing_id_and_prediction_are_warnings(
    schema, reference_stats, clean_df
):
    df = clean_df.drop(
        columns=["customer_id", "prediction"]
    )

    findings = validate_against_schema(
        df, schema, reference_stats
    )

    missing = [
        f for f in findings
        if f.code == "missing_column"
    ]

    assert len(missing) == 2
    assert all(f.severity == "warning" for f in missing)


def test_extra_column_produces_warning(
    schema, reference_stats, clean_df
):
    df = clean_df.copy()
    df["unexpected"] = [1, 2, 3]

    findings = validate_against_schema(
        df, schema, reference_stats
    )

    extra = [
        f for f in findings
        if f.code == "new_column"
    ]

    assert len(extra) == 1
    assert extra[0].column == "unexpected"
    assert extra[0].severity == "warning"


def test_numeric_column_with_5_percent_garbage_is_warning(
    schema, reference_stats
):
    df = pd.DataFrame({
        "Age": [str(i) for i in range(19)] + ["bad"],
        "created_at": ["2025-01-01"] * 20,
        "customer_id": range(20),
        "prediction": [0.5] * 20,
    })

    findings = validate_against_schema(
        df, schema, reference_stats
    )

    mismatch = next(
        f for f in findings
        if f.code == "type_mismatch" and f.column == "Age"
    )

    assert mismatch.severity == "warning"
    assert mismatch.detail["mismatch_rate"] == pytest.approx(0.05)


def test_numeric_column_with_30_percent_garbage_is_error(
    schema, reference_stats
):
    df = pd.DataFrame({
        "Age": ["bad"] * 6 + ["20"] * 14,
        "created_at": ["2025-01-01"] * 20,
        "customer_id": range(20),
        "prediction": [0.5] * 20,
    })

    findings = validate_against_schema(
        df, schema, reference_stats
    )

    mismatch = next(
        f for f in findings
        if f.code == "type_mismatch" and f.column == "Age"
    )

    assert mismatch.severity == "error"
    assert mismatch.detail["mismatch_rate"] == pytest.approx(0.30)


def test_all_null_column_produces_null_rate_jump(
    schema, reference_stats, clean_df
):
    df = clean_df.copy()
    df["Age"] = None

    findings = validate_against_schema(
        df, schema, reference_stats
    )

    jump = next(
        f for f in findings
        if f.code == "null_rate_jump" and f.column == "Age"
    )

    assert jump.severity == "warning"
    assert jump.detail["batch_missing_rate"] == 1.0
    assert jump.detail["reference_missing_rate"] == 0.0


def test_ignore_column_missing_or_garbled_produces_nothing(
    schema, reference_stats, clean_df
):
    df = clean_df.drop(columns=["internal"])

    findings = validate_against_schema(
        df, schema, reference_stats
    )

    assert not any(
        f.column == "internal" for f in findings
    )

    df["internal"] = ["garbage"] * len(df)

    findings = validate_against_schema(
        df, schema, reference_stats
    )

    assert not any(
        f.column == "internal" for f in findings
    )


def test_renamed_column_suggests_original_name(
    schema, reference_stats, clean_df
):
    df = clean_df.rename(columns={"Age": "age"})

    findings = validate_against_schema(
        df, schema, reference_stats
    )

    missing = next(
        f for f in findings
        if f.code == "missing_column" and f.column == "Age"
    )

    assert missing.severity == "error"
    assert "found 'age', did you mean that?" in missing.message


def test_datetime_column_with_invalid_values(
    schema, reference_stats, clean_df
):
    df = clean_df.copy()
    df["created_at"] = [
        "2025-01-01",
        "invalid",
        "2025-01-03",
    ]

    findings = validate_against_schema(
        df, schema, reference_stats
    )

    mismatch = next(
        f for f in findings
        if f.code == "type_mismatch"
        and f.column == "created_at"
    )

    assert mismatch.severity == "warning"
    assert mismatch.detail["mismatch_rate"] == pytest.approx(1 / 3)


def test_time_column_with_any_unparseable_value_is_error():
    schema = [
        SchemaColumn(
            "timestamp",
            ColumnDtype.datetime,
            ColumnRole.time,
        ),
    ]

    df = pd.DataFrame({
        "timestamp": [
            "2024-01-05",
            "2024-01-06T10:00:00",
            "Jan 7 2024",
            "invalid",
        ]
    })

    findings = validate_against_schema(df, schema, {})

    assert len(findings) == 1
    assert findings[0].code == "type_mismatch"
    assert findings[0].severity == "error"
    assert findings[0].detail["mismatch_rate"] == 0.25



def test_mixed_datetime_formats_are_accepted():
    schema = [
        SchemaColumn(
            "timestamp",
            ColumnDtype.datetime,
            ColumnRole.time,
        ),
    ]

    df = pd.DataFrame({
        "timestamp": [
            "2024-01-05",
            "2024-01-06T10:00:00",
            "Jan 7 2024",
            "2024-01-08",
        ]
    })

    findings = validate_against_schema(df, schema, {})

    assert findings == []