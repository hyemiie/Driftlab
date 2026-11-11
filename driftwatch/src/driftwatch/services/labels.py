"""Ingestion for POST /v1/labels (delayed ground truth).

Rules:
  * a label attaches to an existing prediction by (project, row_id)
  * unknown row_id -> status "unknown_id" (the prediction may not have arrived yet; client retries)
  * same label re-sent -> "unchanged"; label_received_at is NOT bumped, so retries are harmless
  * different label -> overwritten, label_received_at refreshed, reported as a finding
  * the same row_id twice in one request -> the LAST one wins, earlier ones are "superseded"
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Sequence

from sqlalchemy import bindparam, select, update
from sqlalchemy.orm import Session

from .ingest import _coerce_value, _Reject
from .models import PredictionRow
from .schemas import FindingOut, LabelIn, LabelResult, LabelsResponse


def ingest_labels(session: Session, project_id: uuid.UUID, task: str, items: Sequence[LabelIn],
                  *, now: datetime | None = None) -> LabelsResponse:
    now = now or datetime.now(timezone.utc)
    results: dict[int, LabelResult] = {}

    # 1. validate; for repeated row_ids the last valid one wins
    parsed: dict[int, tuple[str, float | None, str | None]] = {}
    last: dict[str, int] = {}
    for i, it in enumerate(items):
        if not it.row_id.strip():
            results[i] = LabelResult(row_id=it.row_id, status="rejected", reason="empty_row_id")
            continue
        try:
            num, cat = _coerce_value(it.label, task, "label")
        except _Reject as e:
            results[i] = LabelResult(row_id=it.row_id, status="rejected", reason=str(e))
            continue
        parsed[i] = (it.row_id, num, cat)
        last[it.row_id] = i
    for i, (rid, _, _) in parsed.items():
        if last[rid] != i:
            results[i] = LabelResult(row_id=rid, status="superseded", reason="later_label_in_request")
    winners = {rid: parsed[i] for rid, i in last.items()}

    overwritten = 0
    if winners:
        # 2. lock the target rows in a fixed order (row_id), so concurrent label requests can't deadlock
        existing = {r.row_id: r for r in session.execute(
            select(PredictionRow.row_id, PredictionRow.label_num, PredictionRow.label_cat)
            .where(PredictionRow.project_id == project_id, PredictionRow.row_id.in_(sorted(winners)))
            .order_by(PredictionRow.row_id).with_for_update())}

        # 3. decide per row, with the locks held
        changes = []
        status: dict[str, str] = {}
        for rid, (_, num, cat) in winners.items():
            row = existing.get(rid)
            if row is None:
                status[rid] = "unknown_id"
            elif (row.label_num, row.label_cat) == (num, cat):
                status[rid] = "unchanged"
            else:
                status[rid] = "updated"
                if row.label_num is not None or row.label_cat is not None:
                    overwritten += 1
                changes.append(dict(b_row_id=rid, b_num=num, b_cat=cat))

        # 4. one executemany UPDATE (Core table, not the ORM entity, so no primary-key requirement)
        if changes:
            t = PredictionRow.__table__
            session.execute(
                update(t).where(t.c.project_id == project_id, t.c.row_id == bindparam("b_row_id"))
                .values(label_num=bindparam("b_num"), label_cat=bindparam("b_cat"), label_received_at=now),
                changes)

        for rid, i in last.items():
            reason = "no_prediction_with_this_row_id" if status[rid] == "unknown_id" else None
            results[i] = LabelResult(row_id=rid, status=status[rid], reason=reason)

    ordered = [results[i] for i in range(len(items))]
    findings = []
    if overwritten:
        findings.append(FindingOut(code="labels_overwritten", severity="info", count=overwritten,
                                   message=f"{overwritten} existing labels were replaced with a different value."))
    count = lambda s: sum(r.status == s for r in ordered)  # noqa: E731
    return LabelsResponse(updated=count("updated"), unchanged=count("unchanged"), unknown_id=count("unknown_id"),
                          rejected=count("rejected"), superseded=count("superseded"),
                          results=ordered, findings=findings)