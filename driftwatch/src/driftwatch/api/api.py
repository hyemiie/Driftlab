from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from .deps import ProjectContext, get_project_context, get_session
from .ingest import ingest_predictions
from .labels import ingest_labels
from .schemas import LabelsRequest, LabelsResponse, PredictLogRequest, PredictLogResponse

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