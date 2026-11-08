from pathlib import Path
from uuid import UUID

import pandas as pd

from driftwatch.config import settings

SAMPLE_SIZE = 5_000


def _project_dir(project_id: UUID) -> Path:
    d = Path(settings.data_dir) / str(project_id)
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_reference(project_id: UUID, df: pd.DataFrame) -> Path:
    path = _project_dir(project_id) / "reference.parquet"
    df.to_parquet(path, index=False)
    return path


def load_reference(project_id: UUID) -> pd.DataFrame:
    path = Path(settings.data_dir) / str(project_id) / "reference.parquet"
    if not path.exists():
        raise FileNotFoundError(f"No reference data for project {project_id}")
    return pd.read_parquet(path)


def save_sample(project_id: UUID, df: pd.DataFrame) -> Path:
    path = _project_dir(project_id) / "reference_sample.parquet"
    df.sample(min(len(df), SAMPLE_SIZE), random_state=42).to_parquet(path, index=False)
    return path