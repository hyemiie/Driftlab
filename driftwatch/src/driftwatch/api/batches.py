from dataclasses import asdict
from datetime import datetime
from uuid import UUID

from driftwatch.api.reference import _columns
from driftwatch.services.validation import validate_against_schema
from driftwatch.services.windowing import assign_windows
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from driftwatch.api.deps import get_current_user, get_owned_project
from driftwatch.config import settings
from driftwatch.db.models import (
    ProjectStatus, User,
)
from driftwatch.db.session import get_db
from driftwatch.services.ingestion import read_csv
router = APIRouter(prefix="/projects/{project_id}", tags=["batches"])

import hashlib

router = APIRouter(prefix="/projects/{project_id}", tags=["batches"])

@router.post("/batches", status_code=status.HTTP_201_CREATED)
def upload_batch(
    project_id: UUID,
    file: UploadFile = File(...),
    window_size: str | None = Query(default=None),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    project = get_owned_project(project_id, user, db)

    if project.status != ProjectStatus.ready:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            "Profile the reference data before uploading production data",
        )

    # 1. Read and size-check the uploaded file
    raw = file.file.read(settings.max_upload_bytes + 1)

    if len(raw) > settings.max_upload_bytes:
        raise HTTPException(
            status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            "File too large",
        )

    file_hash = hashlib.sha256(raw).hexdigest()

    # 2. Check for an existing batch with this file hash here.
    # If found, return the existing batch rather than processing it again.

    try:
        # 3. Parse CSV and inspect its schema
        df = read_csv(raw, min_rows=1)
        columns = _columns(db, project.id)

        # TODO: Determine the time column using your SchemaColumn role/dtype.
        # Use None when the project has no time column.
        time_col = ...  # replace with your actual detection logic

        # 4. Determine and validate the requested window size
        requested_window_size = (
            window_size
            if window_size is not None
            else ("day" if time_col is not None else "batch")
        )

        if requested_window_size not in {"day", "week", "batch"}:
            raise HTTPException(
                status_code=400,
                detail="window_size must be day, week, or batch.",
            )

        if time_col is None and requested_window_size != "batch":
            raise HTTPException(
                status_code=400,
                detail="Projects without a time column must use batch windows.",
            )

        if (
            project.window_size is not None
            and project.window_size != requested_window_size
        ):
            raise HTTPException(
                status_code=409,
                detail=(
                    f"Project uses '{project.window_size}' windows. "
                    f"Cannot change to '{requested_window_size}'."
                ),
            )

        # 5. Validate the uploaded data against the reference schema
        reference = load_reference_stats(db, project.id)
        findings = validate_against_schema(df, columns, reference)

        errors = [f for f in findings if f.severity == "error"]
        if errors:
            raise HTTPException(
                status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=[asdict(f) for f in errors],
            )

        # 6. Assign windows after validation
        uploaded_at = datetime.now(timezone.utc)

        df, window_findings = assign_windows(
            df=df,
            time_col=time_col,
            window_size=requested_window_size,
            uploaded_at=uploaded_at,
        )

        warnings = [
            asdict(f)
            for f in findings
            if f.severity == "warning"
        ]
        warnings.extend(asdict(f) for f in window_findings)

        # 7. Set the project's window size only after successful validation
        if project.window_size is None:
            project.window_size = requested_window_size

        # 8. Continue with batch/window/stat persistence here

    except ValueError as e:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            str(e),
        )