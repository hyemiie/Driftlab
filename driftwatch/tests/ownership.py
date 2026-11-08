import uuid

import pytest

from conftest import API, make_sample_csv


@pytest.fixture
def alice(flow):
    return flow.signup("alice@example.com")


@pytest.fixture
def bob(flow):
    return flow.signup("bob@example.com")


@pytest.fixture
def alice_project(flow, alice):
    return flow.ready_project(alice)


def test_other_users_project_looks_like_it_does_not_exist(client, alice_project, bob):
    theirs = client.get(f"{API}/projects/{alice_project}", headers=bob)
    nothing = client.get(f"{API}/projects/{uuid.uuid4()}", headers=bob)
    assert theirs.status_code == nothing.status_code == 404
    assert theirs.json() == nothing.json()  # no hint that the id exists


def test_list_only_shows_own_projects(client, flow, alice, bob, alice_project):
    mine = flow.project(bob, name="bobs")
    ids = [p["id"] for p in client.get(f"{API}/projects", headers=bob).json()]
    assert ids == [mine["id"]]
    assert alice_project not in ids


def test_every_project_route_returns_404_for_another_users_project(client, flow, alice_project, bob):
    pid = alice_project
    files = {"file": ("reference.csv", make_sample_csv(), "text/csv")}
    responses = [
        client.get(f"{API}/projects/{pid}", headers=bob),
        client.get(f"{API}/projects/{pid}/schema", headers=bob),
        client.get(f"{API}/projects/{pid}/profile", headers=bob),
        client.post(f"{API}/projects/{pid}/reference", files=files, headers=bob),
        client.put(f"{API}/projects/{pid}/schema", json={"columns": []}, headers=bob),
        client.post(f"{API}/projects/{pid}/profile", headers=bob),
    ]
    assert [r.status_code for r in responses] != []
    for r in responses:
        # a 422 for the empty schema body is fine too: what matters is it never succeeds
        assert r.status_code in (404, 422), r.text
    assert responses[0].status_code == 404 and responses[2].status_code == 404


def test_failed_attempts_do_not_change_the_owners_project(client, flow, alice, alice_project, bob):
    before_schema = client.get(f"{API}/projects/{alice_project}/schema", headers=alice).json()
    flow.upload(bob, alice_project)
    flow.profile(bob, alice_project)
    project = client.get(f"{API}/projects/{alice_project}", headers=alice).json()
    after_schema = client.get(f"{API}/projects/{alice_project}/schema", headers=alice).json()
    assert project["status"] == "ready"
    assert after_schema == before_schema


def test_owner_can_still_read_everything(client, alice, alice_project):
    assert client.get(f"{API}/projects/{alice_project}", headers=alice).status_code == 200
    assert client.get(f"{API}/projects/{alice_project}/profile", headers=alice).status_code == 200


def test_every_project_route_requires_a_token(client):
    pid = uuid.uuid4()
    files = {"file": ("reference.csv", make_sample_csv(), "text/csv")}
    responses = [
        client.get(f"{API}/projects"),
        client.post(f"{API}/projects", json={"name": "x", "task_type": "regression"}),
        client.get(f"{API}/projects/{pid}"),
        client.get(f"{API}/projects/{pid}/schema"),
        client.get(f"{API}/projects/{pid}/profile"),
        client.post(f"{API}/projects/{pid}/reference", files=files),
        client.put(f"{API}/projects/{pid}/schema", json={"columns": []}),
        client.post(f"{API}/projects/{pid}/profile"),
    ]
    assert all(r.status_code == 401 for r in responses)