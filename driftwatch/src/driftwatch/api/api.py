from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from .deps import ProjectContext, get_project_context, get_session, get_project_access
from .ingest import ingest_predictions
from .labels import ingest_labels
from .schemas import LabelsRequest, LabelsResponse, PredictLogRequest, PredictLogResponse

import uuid
from datetime import datetime
from .performance import compute_window_metrics
from .schemas import (LabelsRequest, LabelsResponse, PerformanceResponse, PredictLogRequest,
                      PredictLogResponse)


router = APIRouter(prefix="/v1", tags=["ingest"])


@router.post("/predict-log", response_model=PredictLogResponse)
def predict_log(req: PredictLogRequest, ctx: ProjectContext = Depends(get_project_context),
                session: Session = Depends(get_session)) -> PredictLogResponse:
    result = ingest_predictions(session, ctx.project_id, ctx.task, ctx.reference, req.items,
                                window_size=ctx.window_size)
    session.commit()
    return result


@router.post("/labels", response_model=LabelsResponse)
def post_labels(req: LabelsRequest, ctx: ProjectContext = Depends(get_project_context),
                session: Session = Depends(get_session)) -> LabelsResponse:
    result = ingest_labels(session, ctx.project_id, ctx.task, req.items)
    session.commit()
    return result


@router.get("/projects/{project_id}/performance", response_model=PerformanceResponse)
def performance(
    project_id: uuid.UUID,
    window_size: str | None = Query(None, pattern="^(day|week)$"),
    since: datetime | None = None,
    until: datetime | None = None,
    model_version: str | None = None,
    min_labels: int = Query(30, ge=1),
    ctx: ProjectContext = Depends(get_project_access),
    session: Session = Depends(get_session),
) -> PerformanceResponse:
    size = window_size or ctx.window_size
    windows = compute_window_metrics(session, ctx.project_id, ctx.task, window_size=size, since=since,
                                     until=until, model_version=model_version, min_labels=min_labels)
    return PerformanceResponse(task=ctx.task, window_size=size, min_labels=min_labels, windows=windows)