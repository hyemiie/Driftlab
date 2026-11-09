from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from driftwatch.api.schemas import ProjectCreate, ProjectRead
from driftwatch.db.models import Project, User
from driftwatch.db.session import get_db
from driftwatch.api.deps import get_current_user, get_owned_project

project_router = APIRouter(prefix="/projects", tags=["projects"])




@project_router.post("", response_model=ProjectRead, status_code=status.HTTP_201_CREATED)
def create_project(
    payload: ProjectCreate,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    project = Project(
        name=payload.name,
        task_type=payload.task_type,
        owner_id=user.id,
    )
    db.add(project)
    db.commit()
    db.refresh(project)
    return project


@project_router.get("", response_model=list[ProjectRead])
def list_projects(
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    return db.scalars(
        select(Project)
        .where(Project.owner_id == user.id)
        .order_by(Project.created_at.desc())
    ).all()


@project_router.get("/{project_id}", response_model=ProjectRead)
def get_project(
    project_id: UUID,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    return get_owned_project(project_id, user, db)

