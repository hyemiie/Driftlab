"""Ingestion for POST /v1/predict-log.

Order of operations (one transaction, committed by the caller):
  1. validate items, drop repeats inside the request
  2. assign windows (reuses windowing.assign_windows)
  3. get-or-create windows, then lock them (ordered by id)
  4. INSERT prediction rows ... ON CONFLICT DO NOTHING RETURNING row_id
  5. profile ONLY the rows that were really inserted (retries never double count)
  6. merge with the stored summaries and upsert
"""
from __future__ import annotations

import math
import uuid
from datetime import datetime, timezone
from functools import reduce
from typing import Any, Mapping, Sequence

import pandas as pd
from sqlalchemy import select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from .models import PredictionRow, Window, WindowColumnSummary
from .schemas import FindingOut, ItemResult, PredictionIn, PredictLogResponse
from .window_profile import (CategoricalSummary, ColumnRef, NumericSummary, WindowSummary,
                             merge_columns, merge_windows, profile_window, summary_from_dict,
                             summary_to_dict)
from .windowing import assign_windows, window_end_for

MIN_EVENT_TIME = datetime(2000, 1, 1, tzinfo=timezone.utc)  # catches epoch-0 / bad units
_SCALAR = (str, int, float, bool, type(None))


class _Reject(Exception):
    pass


def _coerce_value(value: Any, task: str, name: str = "prediction") -> tuple[float | None, str | None]:
    """Shared by predictions and labels. Returns (numeric, categorical); exactly one is set."""
    if task == "regression":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise _Reject(f"{name}_must_be_numeric")
        if not math.isfinite(value):
            raise _Reject(f"non_finite_{name}")
        return float(value), None
    # classification
    if isinstance(value, bool):
        label = str(value)
    elif isinstance(value, float):
        if not math.isfinite(value) or not value.is_integer():
            raise _Reject(f"{name}_must_be_a_class_label")  # a score sent as the label
        label = str(int(value))
    else:
        label = str(value).strip()
    if not label:
        raise _Reject(f"empty_{name}")
    if len(label) > 200:
        raise _Reject(f"{name}_too_long")
    return None, label


def _validate(item: PredictionIn, task: str) -> tuple[float | None, str | None, datetime | None]:
    if not item.row_id.strip():
        raise _Reject("empty_row_id")
    for key, v in item.inputs.items():
        if not isinstance(v, _SCALAR):
            raise _Reject(f"nested_input:{key}")
    num, cat = _coerce_value(item.prediction, task, "prediction")
    ts = item.timestamp
    if ts is not None:
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)  # assumed UTC; reported as a finding
        if ts < MIN_EVENT_TIME:
            raise _Reject("timestamp_out_of_range")
    return num, cat, ts


def _summary_findings(summary: WindowSummary, extra_inputs: dict[str, int]) -> list[FindingOut]:
    out: list[FindingOut] = []
    for name, s in summary.columns.items():
        if s.absent:
            out.append(FindingOut(code="missing_feature", severity="warning", count=summary.n_rows,
                                  message=f"Feature '{name}' was not sent in any item of this batch."))
            continue
        if s.missing:
            out.append(FindingOut(code="missing_values", severity="info", count=s.missing,
                                  message=f"'{name}' was null or absent in {s.missing} of {summary.n_rows} items."))
        if isinstance(s, NumericSummary):
            if s.invalid:
                out.append(FindingOut(code="invalid_values", severity="warning", count=s.invalid,
                                      message=f"'{name}' had {s.invalid} values that are not numbers."))
            if s.n and (s.counts[0] + s.counts[-1]) / s.n > 0.5:
                out.append(FindingOut(code="out_of_reference_range", severity="warning",
                                      count=s.counts[0] + s.counts[-1],
                                      message=f"Over half of '{name}' falls outside the reference range "
                                              "(possible unit or scale change)."))
        elif isinstance(s, CategoricalSummary) and s.unseen_total:
            names = ", ".join(list(s.unseen)[:5])
            out.append(FindingOut(code="unseen_categories", severity="warning", count=s.unseen_total,
                                  message=f"'{name}' had {s.unseen_total} values outside the reference "
                                          f"categories (e.g. {names})."))
    if extra_inputs:
        names = ", ".join(sorted(extra_inputs)[:10])
        out.append(FindingOut(code="unexpected_inputs", severity="info", count=sum(extra_inputs.values()),
                              message=f"Inputs not in the reference were ignored: {names}."))
    return out


def ingest_predictions(
    session: Session,
    project_id: uuid.UUID,
    task: str,
    reference: Mapping[str, ColumnRef],
    items: Sequence[PredictionIn],
    *,
    window_size: str = "day",
    now: datetime | None = None,
) -> PredictLogResponse:
    now = now or datetime.now(timezone.utc)
    results: dict[int, ItemResult] = {}
    findings: list[FindingOut] = []

    # 1. validate, de-duplicate inside the request
    seen: set[str] = set()
    valid: list[dict] = []
    naive = 0
    for i, it in enumerate(items):
        try:
            num, cat, ts = _validate(it, task)
        except _Reject as e:
            results[i] = ItemResult(row_id=it.row_id, status="rejected", reason=str(e))
            continue
        if it.row_id in seen:
            results[i] = ItemResult(row_id=it.row_id, status="duplicate", reason="repeated_in_request")
            continue
        seen.add(it.row_id)
        if it.timestamp is not None and it.timestamp.tzinfo is None:
            naive += 1
        valid.append(dict(idx=i, item=it, num=num, cat=cat, event_time=ts or now))
    if naive:
        findings.append(FindingOut(code="naive_timestamps_assumed_utc", severity="info", count=naive,
                                   message=f"{naive} timestamps had no timezone and were treated as UTC."))

    if valid:
        df = pd.DataFrame({"row_id": [v["item"].row_id for v in valid],
                           "event_time": [v["event_time"] for v in valid]})
        inputs = pd.DataFrame([v["item"].inputs for v in valid], index=df.index)

        # 2. windows
        windowed, wfind = assign_windows(df, "event_time", window_size, now)
        findings += [FindingOut(**vars(f)) for f in wfind]
        for pos in df.index.difference(windowed.index):
            v = valid[pos]
            results[v["idx"]] = ItemResult(row_id=v["item"].row_id, status="rejected", reason="unparseable_timestamp")
        keep = list(windowed.index)

        if keep:
            # 3. get-or-create windows
            starts = [pd.Timestamp(s).to_pydatetime() for s in windowed["window_start"].drop_duplicates()]
            session.execute(
                pg_insert(Window).values([
                    dict(id=uuid.uuid4(), project_id=project_id, window_size=window_size, window_start=s,
                         window_end=window_end_for(pd.Timestamp(s), window_size).to_pydatetime(), n_rows=0)
                    for s in starts
                ]).on_conflict_do_nothing(index_elements=["project_id", "window_size", "window_start"]))
            win_id = {w.window_start: w.id for w in session.execute(
                select(Window).where(Window.project_id == project_id, Window.window_size == window_size,
                                     Window.window_start.in_(starts))).scalars()}

            # Lock every window this batch touches, in id order, BEFORE inserting rows. Inserting
            # first would take FOR KEY SHARE locks through the foreign key, and two requests would
            # then deadlock when each asks for FOR UPDATE.
            session.execute(select(Window.id).where(Window.id.in_(sorted(win_id.values())))
                            .order_by(Window.id).with_for_update())

            # 4. insert rows; sorted by row_id so concurrent requests lock in the same order
            rows = sorted((dict(
                id=uuid.uuid4(), project_id=project_id,
                window_id=win_id[pd.Timestamp(windowed.at[pos, "window_start"]).to_pydatetime()],
                row_id=valid[pos]["item"].row_id,
                event_time=pd.Timestamp(windowed.at[pos, "event_time"]).to_pydatetime(),
                received_at=now, model_version=valid[pos]["item"].model_version,
                prediction_num=valid[pos]["num"], prediction_cat=valid[pos]["cat"],
                prediction_score=valid[pos]["item"].prediction_score) for pos in keep),
                key=lambda r: r["row_id"])
            inserted = {r[0] for r in session.execute(
                pg_insert(PredictionRow).values(rows)
                .on_conflict_do_nothing(index_elements=["project_id", "row_id"])
                .returning(PredictionRow.row_id))}
            for pos in keep:
                v = valid[pos]
                if v["item"].row_id in inserted:
                    results[v["idx"]] = ItemResult(row_id=v["item"].row_id, status="accepted")
                else:
                    results[v["idx"]] = ItemResult(row_id=v["item"].row_id, status="duplicate", reason="already_logged")

            # 5. profile only the newly inserted rows, per window
            new_pos = [p for p in keep if valid[p]["item"].row_id in inserted]
            if new_pos:
                new_ws = windowed.loc[new_pos, "window_start"]
                per_window: dict[uuid.UUID, WindowSummary] = {}
                for start, grp in new_ws.groupby(new_ws):
                    wid = win_id[pd.Timestamp(start).to_pydatetime()]
                    per_window[wid] = profile_window(inputs.loc[grp.index], reference)
                batch_summary = reduce(merge_windows, per_window.values())
                extra: dict[str, int] = {}
                for p in new_pos:
                    for k in valid[p]["item"].inputs:
                        if k not in reference:
                            extra[k] = extra.get(k, 0) + 1
                findings += _summary_findings(batch_summary, extra)

                # 6. merge with stored summaries (windows are already locked) and upsert
                ids = sorted(per_window)
                stored = {(s.window_id, s.column_name): s for s in session.execute(
                    select(WindowColumnSummary).where(WindowColumnSummary.window_id.in_(ids))).scalars()}
                upserts = []
                for wid in ids:
                    for col, new in per_window[wid].columns.items():
                        old = stored.get((wid, col))
                        merged = merge_columns(summary_from_dict(old.summary), new) if old else new
                        upserts.append(dict(window_id=wid, column_name=col, kind=merged.kind,
                                            summary=summary_to_dict(merged)))
                    session.execute(update(Window).where(Window.id == wid)
                                    .values(n_rows=Window.n_rows + per_window[wid].n_rows))
                if upserts:
                    stmt = pg_insert(WindowColumnSummary).values(upserts)
                    session.execute(stmt.on_conflict_do_update(
                        index_elements=["window_id", "column_name"],
                        set_={"kind": stmt.excluded.kind, "summary": stmt.excluded.summary}))

    ordered = [results[i] for i in range(len(items))]
    return PredictLogResponse(
        accepted=sum(r.status == "accepted" for r in ordered),
        duplicates=sum(r.status == "duplicate" for r in ordered),
        rejected=sum(r.status == "rejected" for r in ordered),
        results=ordered, findings=findings)
