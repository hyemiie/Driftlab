from __future__ import annotations
from uuid import UUID

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy import select, create_engine
from sqlalchemy.orm import Session, sessionmaker

from driftwatch.config import settings
from driftwatch.db.models import ApiKey, Project, User
from driftwatch.db.session import get_db


import hashlib
import os
import secrets
import uuid
from dataclasses import dataclass
from typing import Mapping

from fastapi import Depends, Header, HTTPException

# from .models import ApiKey

_engine = None
_Session = None

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login")

def get_owned_project(project_id: UUID, user: User, db: Session) -> Project:
    project = db.scalar(
        select(Project).where(
            Project.id == project_id,
            Project.owner_id == user.id,
        )
    )
    if project is None:
        raise HTTPException(status_code=404, detail="Project not found")
    return project


def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db),
) -> User:
    credentials_error = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    

    try:
        payload = jwt.decode(
            token, settings.jwt_secret, algorithms=[settings.jwt_algorithm]
        )
        user_id = UUID(payload["sub"])
    except (jwt.PyJWTError, KeyError, ValueError):
        raise credentials_error

    user = db.get(User, user_id)
    if user is None:
        raise credentials_error

    return user






def get_session():  # replace with your existing session dependency
    global _engine, _Session
    if _engine is None:
        _engine = create_engine(os.environ["DATABASE_URL"], pool_pre_ping=True)
        _Session = sessionmaker(_engine, expire_on_commit=False)
    with _Session() as session:
        yield session


def new_api_key() -> tuple[str, str, str]:
    """Returns (plaintext, sha256 hash, display prefix). Show the plaintext once; store only the hash."""
    plaintext = "dw_" + secrets.token_urlsafe(32)
    return plaintext, hashlib.sha256(plaintext.encode()).hexdigest(), plaintext[:11]


def get_project_id(x_api_key: str = Header(...), session: Session = Depends(get_session)) -> uuid.UUID:
    digest = hashlib.sha256(x_api_key.encode()).hexdigest()
    row = session.execute(
        select(ApiKey).where(ApiKey.key_hash == digest, ApiKey.revoked_at.is_(None))
    ).scalar_one_or_none()
    if row is None:
        raise HTTPException(401, "Invalid or revoked API key.")
    return row.project_id


@dataclass
class ProjectContext:
    project_id: uuid.UUID
    task: str                        # "regression" | "classification"
    window_size: str                 # "day" | "week"
    reference: Mapping[str, ColumnRef]


def get_project_context(project_id: uuid.UUID = Depends(get_project_id),
                        session: Session = Depends(get_session)) -> ProjectContext:
    """WIRE THIS UP: load the project's task type, window size and reference bin edges /
    categories from reference_profiles + column_stats, wrapped in NumericRef / CategoricalRef."""
    raise NotImplementedError

def get_project_access(project_id: uuid.UUID, session: Session = Depends(get_session)) -> ProjectContext:
    """WIRE THIS UP: dashboard routes use your normal user auth (JWT), not the API key.
    Check the signed-in user may read this project, then return its ProjectContext."""
    raise NotImplementedError