"""Per-window model performance from prediction_rows, computed on demand.

Computed live (SQL aggregates) instead of cached: labels keep arriving for old windows, so a
stored metric would go stale. Every window carries label coverage and a quality flag so a
number from 5% labeled rows is never mistaken for a number from 95%.
"""
from __future__ import annotations

import math
import uuid
from collections import defaultdict
from datetime import datetime, timezone

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from .models import PredictionRow as P
from .models import Window
from .schemas import ClassMetrics, WindowMetrics

MAX_CLASSES = 20  # per-class table / confusion matrix are dropped beyond this


def _utc(dt: datetime | None) -> datetime | None:
    return dt if dt is None or dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _classification(conf: dict[str, dict[str, int]]) -> dict:
    """conf[actual][predicted] -> accuracy, macro scores, per-class table."""
    classes = sorted({a for a in conf} | {p for row in conf.values() for p in row})
    total = sum(sum(row.values()) for row in conf.values())
    correct = sum(conf[c].get(c, 0) for c in conf)
    per_class: list[ClassMetrics] = []
    for c in classes:
        tp = conf.get(c, {}).get(c, 0)
        fn = sum(conf.get(c, {}).values()) - tp
        fp = sum(row.get(c, 0) for row in conf.values()) - tp
        per_class.append(ClassMetrics(
            label=c,
            precision=tp / (tp + fp) if tp + fp else None,
            recall=tp / (tp + fn) if tp + fn else None,
            f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
            support=tp + fn))
    present = [m for m in per_class if m.support > 0]  # macro over classes that really occur in the labels

    def macro(attr: str) -> float | None:
        return sum(getattr(m, attr) or 0.0 for m in present) / len(present) if present else None

    ranked = sorted(per_class, key=lambda m: (-m.support, m.label))
    small = len(classes) <= MAX_CLASSES
    return dict(
        accuracy=correct / total if total else None,
        macro_precision=macro("precision"), macro_recall=macro("recall"), macro_f1=macro("f1"),
        per_class=ranked if small else None,
        confusion={a: dict(row) for a, row in conf.items()} if small else None)


def compute_window_metrics(
    session: Session,
    project_id: uuid.UUID,
    task: str,
    *,
    window_size: str = "day",
    since: datetime | None = None,
    until: datetime | None = None,
    model_version: str | None = None,
    min_labels: int = 30,
) -> list[WindowMetrics]:
    reg = task == "regression"
    label_col = P.label_num if reg else P.label_cat
    delay = func.extract("epoch", P.label_received_at - P.received_at)

    where = [P.project_id == project_id, Window.window_size == window_size]
    if since:
        where.append(Window.window_start >= _utc(since))
    if until:
        where.append(Window.window_start < _utc(until))
    if model_version is not None:
        where.append(P.model_version == model_version)

    cols = [
        Window.id, Window.window_start, Window.window_end,
        func.count().label("n_pred"),
        func.count(label_col).label("n_lab"),
        func.count(func.distinct(P.model_version)).label("n_versions"),
        func.percentile_cont(0.5).within_group(delay).label("d50"),
        func.percentile_cont(0.9).within_group(delay).label("d90"),
    ]
    if reg:
        err = P.prediction_num - P.label_num
        cols += [func.sum(func.abs(err)).label("s_abs"), func.sum(err * err).label("s_sq"),
                 func.sum(err).label("s_err"), func.var_pop(P.label_num).label("var_y")]
    rows = session.execute(
        select(*cols).select_from(P).join(Window, Window.id == P.window_id)
        .where(*where).group_by(Window.id).order_by(Window.window_start)).all()

    confusions: dict[uuid.UUID, dict[str, dict[str, int]]] = defaultdict(lambda: defaultdict(dict))
    if not reg and rows:
        q = (select(P.window_id, P.label_cat, P.prediction_cat, func.count())
             .where(P.project_id == project_id, P.label_cat.is_not(None),
                    P.window_id.in_([r.id for r in rows]),
                    *([P.model_version == model_version] if model_version is not None else []))
             .group_by(P.window_id, P.label_cat, P.prediction_cat))
        for wid, actual, predicted, n in session.execute(q):
            confusions[wid][actual][predicted] = n

    out: list[WindowMetrics] = []
    for r in rows:
        quality = "no_labels" if r.n_lab == 0 else "low_labels" if r.n_lab < min_labels else "ok"
        m = WindowMetrics(
            window_id=r.id, window_start=r.window_start, window_end=r.window_end,
            n_predictions=r.n_pred, n_labeled=r.n_lab,
            label_coverage=r.n_lab / r.n_pred if r.n_pred else 0.0,
            label_delay_p50_s=float(r.d50) if r.d50 is not None else None,
            label_delay_p90_s=float(r.d90) if r.d90 is not None else None,
            n_model_versions=r.n_versions, quality=quality)
        if r.n_lab:
            if reg:
                m.mae = float(r.s_abs) / r.n_lab
                m.rmse = math.sqrt(float(r.s_sq) / r.n_lab)
                m.bias = float(r.s_err) / r.n_lab
                sst = float(r.var_y) * r.n_lab
                m.r2 = 1 - float(r.s_sq) / sst if r.n_lab >= 2 and sst > 1e-12 else None
            else:
                for k, v in _classification(confusions[r.id]).items():
                    setattr(m, k, v)
        out.append(m)
    return out