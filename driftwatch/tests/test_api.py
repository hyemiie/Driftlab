import uuid

import pytest

from conftest import API, make_sample_csv

PASSWORD = "correct-horse-battery"


# ---------------- auth ----------------
def test_signup_returns_user_without_password(client):
    r = client.post(f"{API}/auth/signup", json={"email": "Yemi@Example.com", "password": PASSWORD})
    assert r.status_code == 201
    body = r.json()
    assert body["email"] == "yemi@example.com"  # lowercased
    assert "password" not in body and "hashed_password" not in body


def test_duplicate_email_is_rejected_case_insensitively(client):
    client.post(f"{API}/auth/signup", json={"email": "a@example.com", "password": PASSWORD})
    r = client.post(f"{API}/auth/signup", json={"email": "A@EXAMPLE.COM", "password": PASSWORD})
    assert r.status_code == 409


@pytest.mark.parametrize(
    "payload",
    [
        {"email": "not-an-email", "password": PASSWORD},
        {"email": "a@example.com", "password": "short"},
    ],
)
def test_signup_validation(client, payload):
    assert client.post(f"{API}/auth/signup", json=payload).status_code == 422


def test_login_and_me(client, flow):
    h = flow.signup("me@example.com")
    r = client.get(f"{API}/auth/me", headers=h)
    assert r.status_code == 200 and r.json()["email"] == "me@example.com"


def test_login_email_is_case_insensitive(client, flow):
    flow.signup("me@example.com")
    r = client.post(f"{API}/auth/login", data={"username": "ME@example.com", "password": PASSWORD})
    assert r.status_code == 200


def test_wrong_password_and_unknown_email_look_the_same(client, flow):
    flow.signup("me@example.com")
    wrong_pw = client.post(f"{API}/auth/login", data={"username": "me@example.com", "password": "nope-nope-nope"})
    no_user = client.post(f"{API}/auth/login", data={"username": "ghost@example.com", "password": PASSWORD})
    assert wrong_pw.status_code == no_user.status_code == 401
    assert wrong_pw.json() == no_user.json()


def test_me_requires_a_valid_token(client):
    assert client.get(f"{API}/auth/me").status_code == 401
    r = client.get(f"{API}/auth/me", headers={"Authorization": "Bearer garbage"})
    assert r.status_code == 401


# ---------------- projects ----------------
def test_create_project(flow, headers):
    p = flow.project(headers, name="House prices", task_type="regression")
    assert p["name"] == "House prices"
    assert p["task_type"] == "regression"
    assert p["status"] == "created"
    uuid.UUID(p["id"])


@pytest.mark.parametrize(
    "payload",
    [
        {"name": "ok", "task_type": "clustering"},
        {"name": "", "task_type": "regression"},
        {"name": "   ", "task_type": "regression"},
        {"name": "x" * 201, "task_type": "regression"},
    ],
)
def test_create_project_validation(client, headers, payload):
    assert client.post(f"{API}/projects", json=payload, headers=headers).status_code == 422


def test_list_and_get_projects(client, flow, headers):
    first = flow.project(headers, name="first")
    second = flow.project(headers, name="second")
    listed = client.get(f"{API}/projects", headers=headers).json()
    assert [p["id"] for p in listed] == [second["id"], first["id"]]  # newest first
    r = client.get(f"{API}/projects/{first['id']}", headers=headers)
    assert r.status_code == 200 and r.json()["name"] == "first"


def test_get_missing_project_is_404_and_bad_id_is_422(client, headers):
    assert client.get(f"{API}/projects/{uuid.uuid4()}", headers=headers).status_code == 404
    assert client.get(f"{API}/projects/not-a-uuid", headers=headers).status_code == 422


# ---------------- reference upload ----------------
def test_upload_proposes_a_schema_and_moves_to_schema_pending(client, flow, headers):
    p = flow.project(headers)
    r = flow.upload(headers, p["id"])
    assert r.status_code == 200
    proposal = {c["name"]: c for c in r.json()}
    assert proposal["customer_id"]["dtype"] == "id" and proposal["customer_id"]["role"] == "id"
    assert proposal["age"]["dtype"] == "numeric"
    assert proposal["plan"]["dtype"] == "categorical"
    assert proposal["churned"]["dtype"] == "categorical"
    assert all(c["reason"] for c in proposal.values())
    status = client.get(f"{API}/projects/{p['id']}", headers=headers).json()["status"]
    assert status == "schema_pending"


def test_reupload_replaces_the_proposed_schema(client, flow, headers):
    p = flow.project(headers)
    flow.upload(headers, p["id"])
    flow.upload(headers, p["id"])
    schema = client.get(f"{API}/projects/{p['id']}/schema", headers=headers).json()
    assert len(schema) == 6  # not 12


@pytest.mark.parametrize(
    "content, expected",
    [
        (b"", "empty"),
        (b"a,b,a\n" + b"1,2,3\n" * 40, "Duplicate"),
        (b"a,b\n1,2\n3,4\n", "rows"),
    ],
)
def test_bad_uploads_are_rejected_with_readable_errors(flow, headers, content, expected):
    p = flow.project(headers)
    r = flow.upload(headers, p["id"], content)
    assert r.status_code == 422
    assert expected in r.text


def test_oversized_upload_is_rejected(flow, headers, monkeypatch):
    from driftwatch.config import settings

    monkeypatch.setattr(settings, "max_upload_bytes", 100)
    p = flow.project(headers)
    assert flow.upload(headers, p["id"], make_sample_csv()).status_code == 413


# ---------------- schema confirmation ----------------
def _pending_project(flow, headers, task_type="classification"):
    p = flow.project(headers, task_type=task_type)
    proposal = flow.upload(headers, p["id"]).json()
    return p["id"], proposal


def test_valid_schema_is_saved(client, flow, headers):
    pid, proposal = _pending_project(flow, headers)
    r = flow.put_schema(headers, pid, flow.schema_body(proposal))
    assert r.status_code == 200
    roles = {c["name"]: c["role"] for c in r.json()}
    assert roles["churned"] == "target" and roles["customer_id"] == "id"
    assert client.get(f"{API}/projects/{pid}", headers=headers).json()["status"] == "schema_pending"


def test_schema_needs_exactly_one_target(flow, headers):
    pid, proposal = _pending_project(flow, headers)
    no_target = flow.schema_body(proposal, target="nonexistent")
    assert flow.put_schema(headers, pid, no_target).status_code == 422

    two_targets = flow.schema_body(proposal)
    next(c for c in two_targets["columns"] if c["name"] == "plan")["role"] = "target"
    assert flow.put_schema(headers, pid, two_targets).status_code == 422


def test_schema_must_match_uploaded_columns(flow, headers):
    pid, proposal = _pending_project(flow, headers)
    missing = flow.schema_body(proposal)
    missing["columns"].pop()
    assert flow.put_schema(headers, pid, missing).status_code == 422

    unknown = flow.schema_body(proposal)
    unknown["columns"].append({"name": "ghost", "dtype": "numeric", "role": "feature"})
    assert flow.put_schema(headers, pid, unknown).status_code == 422

    duplicate = flow.schema_body(proposal)
    duplicate["columns"].append(dict(duplicate["columns"][0]))
    assert flow.put_schema(headers, pid, duplicate).status_code == 422


def test_target_dtype_must_fit_the_task_type(flow, headers):
    pid, proposal = _pending_project(flow, headers, task_type="regression")
    categorical_target = flow.schema_body(proposal, target="churned")
    assert flow.put_schema(headers, pid, categorical_target).status_code == 422
    numeric_target = flow.schema_body(proposal, target="monthly_spend")
    assert flow.put_schema(headers, pid, numeric_target).status_code == 200

    pid, proposal = _pending_project(flow, headers, task_type="classification")
    wrong = flow.schema_body(proposal, target="monthly_spend")
    assert flow.put_schema(headers, pid, wrong).status_code == 422


def test_role_and_dtype_must_agree(flow, headers):
    pid, proposal = _pending_project(flow, headers)
    body = flow.schema_body(proposal)
    next(c for c in body["columns"] if c["name"] == "customer_id")["role"] = "time"
    assert flow.put_schema(headers, pid, body).status_code == 422  # time needs a datetime dtype


# ---------------- profiling ----------------
def test_cannot_profile_before_the_schema_is_confirmed(flow, headers):
    """Fails until create_profile checks that exactly one target is set (see notes)."""
    pid, _ = _pending_project(flow, headers)
    assert flow.profile(headers, pid).status_code == 409


def test_full_flow_produces_a_ready_project_with_stats(client, flow, headers):
    pid = flow.ready_project(headers)
    assert client.get(f"{API}/projects/{pid}", headers=headers).json()["status"] == "ready"

    r = client.get(f"{API}/projects/{pid}/profile", headers=headers)
    assert r.status_code == 200
    profile = r.json()
    assert profile["n_rows"] == 120
    stats = {c["column_name"]: c["stats"] for c in profile["columns"]}
    assert set(stats) == {"customer_id", "age", "plan", "tenure_months", "monthly_spend", "churned"}
    assert stats["age"]["kind"] == "numeric" and stats["age"]["missing_rate"] > 0
    assert stats["churned"]["role"] == "target"
    assert sum(c["proportion"] for c in stats["churned"]["categories"]) == pytest.approx(1.0)
    assert sum(stats["age"]["bin_counts"]) == stats["age"]["count"]


def test_ignored_columns_are_not_profiled(client, flow, headers):
    pid, proposal = _pending_project(flow, headers)
    body = flow.schema_body(proposal, overrides={"plan": "ignore"})
    assert flow.put_schema(headers, pid, body).status_code == 200
    assert flow.profile(headers, pid).status_code == 201
    columns = client.get(f"{API}/projects/{pid}/profile", headers=headers).json()["columns"]
    names = {c["column_name"] for c in columns}
    assert "plan" not in names and len(names) == 5


def test_profile_before_it_exists_is_404(client, flow, headers):
    p = flow.project(headers)
    assert client.get(f"{API}/projects/{p['id']}/profile", headers=headers).status_code == 404


def test_ready_projects_are_locked(client, flow, headers):
    pid = flow.ready_project(headers)
    assert flow.profile(headers, pid).status_code == 409
    assert flow.upload(headers, pid).status_code == 409
    schema = client.get(f"{API}/projects/{pid}/schema", headers=headers).json()
    body = {"columns": [{k: c[k] for k in ("name", "dtype", "role")} for c in schema]}
    assert flow.put_schema(headers, pid, body).status_code == 409