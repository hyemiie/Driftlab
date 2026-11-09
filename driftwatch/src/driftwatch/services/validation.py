
from dataclasses import dataclass, field
from difflib import get_close_matches
from typing import Literal

import pandas as pd

from driftwatch.db.models import ColumnDtype, ColumnRole
from driftwatch.services.profiling import to_datetime


# Validation thresholds
NULL_RATE_JUMP_THRESHOLD = 0.20
TYPE_MISMATCH_ERROR_THRESHOLD = 0.20
SUGGESTION_CUTOFF = 0.6


@dataclass
class Finding:
    severity: Literal["error", "warning"]
    code: str
    column: str | None
    message: str
    detail: dict = field(default_factory=dict)


def _suggest_column(
    expected: str,
    unknown_columns: set[str],
) -> str | None:
    """Find a case-insensitive or fuzzy column-name match."""

    lowered = {
        name.lower().strip(): name
        for name in unknown_columns
    }

    match = lowered.get(expected.lower().strip())

    if match:
        return match

    matches = get_close_matches(
        expected,
        list(unknown_columns),
        n=1,
        cutoff=SUGGESTION_CUTOFF,
    )

    return matches[0] if matches else None


def _check_missing(
    df: pd.DataFrame,
    columns: list,
    schema_names: set[str],
) -> list[Finding]:
    findings = []
    batch_names = set(df.columns)
    unknown_columns = batch_names - schema_names

    for column in columns:
        if column.dtype == ColumnDtype.ignore:
            continue

        if column.name in batch_names:
            continue

        if column.role in (
            ColumnRole.feature,
            ColumnRole.time,
        ):
            severity = "error"

        elif column.role in (
            ColumnRole.id,
            ColumnRole.prediction,
        ):
            severity = "warning"

        else:
            # Targets can arrive later.
            continue

        suggested = _suggest_column(
            column.name,
            unknown_columns,
        )

        message = f"Missing column '{column.name}'"

        if suggested:
            message += (
                f" (found '{suggested}', "
                "did you mean that?)"
            )

        findings.append(
            Finding(
                severity=severity,
                code="missing_column",
                column=column.name,
                message=message,
                detail={
                    "expected_column": column.name,
                    "suggested_column": suggested,
                },
            )
        )

    return findings


def _check_new(
    df: pd.DataFrame,
    schema_names: set[str],
) -> list[Finding]:
    findings = []

    for name in df.columns:
        if name in schema_names:
            continue

        findings.append(
            Finding(
                severity="warning",
                code="new_column",
                column=name,
                message=(
                    f"Column '{name}' is not in the schema "
                    "and will be ignored."
                ),
                detail={"column": name},
            )
        )

    return findings


def _check_types(
    df: pd.DataFrame,
    columns: list,
) -> list[Finding]:
    findings = []

    for column in columns:
        if column.dtype == ColumnDtype.ignore:
            continue

        if column.name not in df.columns:
            continue

        series = df[column.name]

        if column.dtype == ColumnDtype.numeric:
            parsed = pd.to_numeric(
                series,
                errors="coerce",
            )

        elif column.dtype == ColumnDtype.datetime:
            parsed = to_datetime(series)

        else:
            # Categorical and text columns are not parsed.
            continue

        non_null = series.notna()
        non_null_count = int(non_null.sum())

        if non_null_count == 0:
            continue

        unparseable_count = int(
            parsed[non_null].isna().sum()
        )

        if unparseable_count == 0:
            continue

        mismatch_rate = (
            unparseable_count / non_null_count
        )

        # Time columns must be valid for window assignment.
        if column.role == ColumnRole.time:
            severity = "error"
        else:
            severity = (
                "error"
                if mismatch_rate >= TYPE_MISMATCH_ERROR_THRESHOLD
                else "warning"
            )

        dtype_name = column.dtype.value

        findings.append(
            Finding(
                severity=severity,
                code="type_mismatch",
                column=column.name,
                message=(
                    f"Column '{column.name}' contains "
                    f"{unparseable_count}/{non_null_count} "
                    f"non-null values that cannot be parsed "
                    f"as {dtype_name} ({mismatch_rate:.1%})."
                ),
                detail={
                    "dtype": dtype_name,
                    "unparseable_count": unparseable_count,
                    "non_null_count": non_null_count,
                    "mismatch_rate": float(mismatch_rate),
                },
            )
        )

    return findings


def _check_nulls(
    df: pd.DataFrame,
    columns: list,
    reference_stats: dict,
) -> list[Finding]:
    findings = []

    for column in columns:
        if column.dtype == ColumnDtype.ignore:
            continue

        if column.name not in df.columns:
            continue

        reference = reference_stats.get(column.name)

        if reference is None:
            continue

        reference_rate = reference.get("missing_rate")

        if reference_rate is None:
            continue

        batch_rate = float(
            df[column.name].isna().mean()
        )

        increase = batch_rate - reference_rate

        if increase <= NULL_RATE_JUMP_THRESHOLD:
            continue

        findings.append(
            Finding(
                severity="warning",
                code="null_rate_jump",
                column=column.name,
                message=(
                    f"Column '{column.name}' has a missing "
                    f"value rate of {batch_rate:.1%}, compared "
                    f"with the reference rate of "
                    f"{reference_rate:.1%}."
                ),
                detail={
                    "batch_missing_rate": batch_rate,
                    "reference_missing_rate": float(reference_rate),
                    "absolute_increase": float(increase),
                    "threshold": NULL_RATE_JUMP_THRESHOLD,
                },
            )
        )

    return findings


def validate_against_schema(
    df: pd.DataFrame,
    columns: list,
    reference_stats: dict,
) -> list[Finding]:
    """
    Validate a batch against its reference schema.

    Returns findings rather than raising exceptions.

    Errors reject the batch at the route level.
    Warnings are accepted and saved with the batch.
    """

    schema_names = {
        column.name
        for column in columns
    }

    findings = []

    findings.extend(
        _check_missing(df, columns, schema_names)
    )

    findings.extend(
        _check_new(df, schema_names)
    )

    findings.extend(
        _check_types(df, columns)
    )

    findings.extend(
        _check_nulls(df, columns, reference_stats)
    )

    return findings
