import uvicorn
from fastapi import APIRouter, Depends, FastAPI
from sqlalchemy import text
from sqlalchemy.orm import Session
from driftwatch.api.projects import project_router
from driftwatch.api.batches import router as batch_router
from driftwatch.db.session import get_db
from driftwatch.api import auth, reference

app = FastAPI(title="Driftwatch")

health_router = APIRouter(prefix="/api/v1", tags=["main"])


@health_router.get("/health")
def health(db: Session = Depends(get_db)):
    db.execute(text("SELECT 1"))
    return {"status": "ok"}


app.include_router(health_router)
app.include_router(project_router)

app.include_router(auth.router, prefix="/api/v1")
app.include_router(reference.router, prefix="/api/v1")
app.include_router(batch_router, prefix="/api/v1")


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=False)