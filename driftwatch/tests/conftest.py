"""Shared test setup.

Pure tests (ingestion, schema inference, profiling) need nothing.
API tests need a throwaway Postgres database: set TEST_DATABASE_URL, and the
database name must end in "_test" (tests wipe all tables).
"""
import os
import tempfile

from dotenv import load_dotenv

load_dotenv()  # so TEST_DATABASE_URL can live in .env

# Must be set before driftwatch.config is imported anywhere.
TEST_DATABASE_URL = os.environ.get("TEST_DATABASE_URL")
os.environ["JWT_SECRET"] = "test-secret-not-for-production"
os.environ["DATA_DIR"] = tempfile.mkdtemp(prefix="driftwatch_test_data_")
os.environ["DATABASE_URL"] = (
    TEST_DATABASE_URL or "postgresql+psycopg2://unused:unused@localhost:1/unused_test"
)

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url
from sqlalchemy.orm import sessionmaker

API = "/api/v1"  # change if your routers are mounted differently
TABLES = "column_stats, reference_profiles, schema_columns, projects, users"


def make_sample_csv(n: int = 120, seed: int = 0) -> bytes:
    """A small churn-style dataset with one of each column kind."""
    rng = np.random.default_rng(seed)
    age = rng.normal(40, 12, n).round(1)
    age[rng.choice(n, size=n // 10, replace=False)] = np.nan
    df = pd.DataFrame(
        {
            "customer_id": np.arange(1, n + 1),
            "age": age,
            "plan": rng.choice(["basic", "pro", "team"], n),
            "tenure_months": rng.integers(0, 61, n),
            "monthly_spend": rng.normal(80, 25, n).round(2),
            "churned": rng.integers(0, 2, n),
        }
    )
    return df.to_csv(index=False).encode()


@pytest.fixture(scope="session")
def engine():
    if not TEST_DATABASE_URL:
        pytest.skip("TEST_DATABASE_URL not set; skipping API tests")
    url = make_url(TEST_DATABASE_URL)
    if "neon.tech" in (url.host or "") or not (url.database or "").endswith("_test"):
        pytest.exit(
            "Refusing to run: TEST_DATABASE_URL must point at a database whose "
            "name ends in '_test' (never Neon).",
            returncode=2,
        )
    from driftwatch.db.models import Base

    eng = create_engine(url)
    Base.metadata.drop_all(eng)
    Base.metadata.create_all(eng)
    yield eng
    eng.dispose()


@pytest.fixture
def client(engine):
    from driftwatch.db.session import get_db
    from driftwatch.main import app

    TestSession = sessionmaker(bind=engine)

    def override_get_db():
        db = TestSession()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()
    with engine.begin() as conn:  # clean slate for the next test
        conn.execute(text(f"TRUNCATE {TABLES} RESTART IDENTITY CASCADE"))


class Flow:
    """Small helpers so tests read like the user flow."""

    def __init__(self, client):
        self.client = client

    def signup(self, email="user@example.com", password="correct-horse-battery"):
        r = self.client.post(f"{API}/auth/signup", json={"email": email, "password": password})
        assert r.status_code == 201, r.text
        r = self.client.post(f"{API}/auth/login", data={"username": email, "password": password})
        assert r.status_code == 200, r.text
        return {"Authorization": f"Bearer {r.json()['access_token']}"}

    def project(self, headers, name="Churn model", task_type="classification"):
        r = self.client.post(
            f"{API}/projects", json={"name": name, "task_type": task_type}, headers=headers
        )
        assert r.status_code == 201, r.text
        return r.json()

    def upload(self, headers, project_id, content=None):
        content = make_sample_csv() if content is None else content
        return self.client.post(
            f"{API}/projects/{project_id}/reference",
            files={"file": ("reference.csv", content, "text/csv")},
            headers=headers,
        )

    @staticmethod
    def schema_body(proposal, target="churned", overrides=None):
        """Turn an upload response into a PUT /schema body."""
        overrides = overrides or {}
        return {
            "columns": [
                {
                    "name": c["name"],
                    "dtype": overrides.get(c["name"], c["dtype"]),
                    "role": "target" if c["name"] == target else c["role"],
                }
                for c in proposal
            ]
        }

    def put_schema(self, headers, project_id, body):
        return self.client.put(f"{API}/projects/{project_id}/schema", json=body, headers=headers)

    def profile(self, headers, project_id):
        return self.client.post(f"{API}/projects/{project_id}/profile", headers=headers)

    def ready_project(self, headers):
        """Create, upload, confirm schema, profile. Returns the project id."""
        p = self.project(headers)
        r = self.upload(headers, p["id"])
        assert r.status_code == 200, r.text
        r = self.put_schema(headers, p["id"], self.schema_body(r.json()))
        assert r.status_code == 200, r.text
        r = self.profile(headers, p["id"])
        assert r.status_code == 201, r.text
        return p["id"]


@pytest.fixture
def flow(client):
    return Flow(client)


@pytest.fixture
def headers(flow):
    return flow.signup()