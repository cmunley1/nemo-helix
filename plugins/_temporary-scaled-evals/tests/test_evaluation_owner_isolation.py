# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Actual route/repository SQL; synthetic admission and task Entity Store only."""

import json
import os
from collections.abc import Awaitable, Callable, Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime
from threading import Event
from typing import Literal, Never
from uuid import uuid4

import httpx
import pytest

pytest.importorskip("scaled_evals")
pytest.importorskip("psycopg")
import psycopg
from fake_entity_store import FakeEntityStore
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient
from nemo_helix_plugin.auth import AuthContext
from nemo_scaled_evals_plugin.authz import stamp_router_authz
from nemo_scaled_evals_plugin.migrations import apply_sql
from psycopg.conninfo import make_conninfo
from psycopg.rows import DictRow, dict_row
from psycopg.sql import SQL, Identifier
from pydantic import BaseModel
from scaled_evals.api import auth, entity_store
from scaled_evals.api.db import Database, get_conn, get_stream_database_factory
from scaled_evals.api.repositories.access import ResourceAccess
from scaled_evals.api.repositories.base_repository import Conflict
from scaled_evals.api.repositories.benchmark_run_repository import BenchmarkRunRepository
from scaled_evals.api.repositories.evaluation_repository import EvaluationRepository
from scaled_evals.api.routers import benchmark_runs, evaluations
from scaled_evals.api.runnability import caller_secret_readable
from scaled_evals.api.schemas.common import ListEnvelope
from scaled_evals.api.schemas.evaluations import Evaluation, EvaluationResponse
from scaled_evals.api.settings import settings
from scaled_evals.api.task_store import TaskStore
from scaled_evals.api.tenancy import caller_is_admin
from task_fixtures import seed_task_revision

USER_A = "synthetic-owner-a"
USER_B = "synthetic-owner-b"
BASE = "/apis/scaled-evals/v1/evaluations"
AUTH_CONTEXT: ContextVar[AuthContext | None] = ContextVar("repro_auth", default=None)
Operation = Literal["get", "cancel", "delete"]


class RowState(BaseModel):
    id: str
    owner_id: str | None
    status: str
    deleted_at: datetime | None


@dataclass
class Harness:
    client: TestClient
    connection: psycopg.Connection[DictRow]
    admin_probes: list[str]

    def row(self, evaluation_id: str) -> RowState:
        result = self.connection.execute(
            "SELECT id, owner_id, status::text, deleted_at FROM evaluations WHERE id = %s", (evaluation_id,)
        ).fetchone()
        assert result is not None
        return RowState.model_validate(result)

    def request(self, operation: Operation, evaluation_id: str, user: str, *, admin: bool = False) -> httpx.Response:
        headers = {"x-repro-user": user, "x-repro-admin": str(admin).lower()}
        before = self.row(evaluation_id) if evaluation_id != "ev_missing" else None
        if operation == "get":
            response = self.client.get(f"{BASE}/{evaluation_id}", headers=headers)
        elif operation == "cancel":
            response = self.client.post(f"{BASE}/{evaluation_id}/cancel", headers=headers)
        else:
            response = self.client.delete(f"{BASE}/{evaluation_id}", headers=headers)
        after = self.row(evaluation_id) if evaluation_id != "ev_missing" else None
        print(
            "REQUEST",
            json.dumps(
                {
                    "user": user,
                    "admin": admin,
                    "operation": operation,
                    "id": evaluation_id,
                    "http": response.status_code,
                    "body": response.json(),
                    "before": before.model_dump(mode="json") if before else None,
                    "after": after.model_dump(mode="json") if after else None,
                },
                default=str,
                sort_keys=True,
            ),
        )
        return response


@pytest.fixture(scope="module")
def postgres_dsn() -> Iterator[str]:
    admin_dsn = os.environ.get("SCALED_EVALS_TEST_DATABASE_URL")
    if not admin_dsn:
        pytest.skip("SCALED_EVALS_TEST_DATABASE_URL is not set")
    scratch = f"se_owner_test_{uuid4().hex[:12]}"
    dsn = make_conninfo(admin_dsn, dbname=scratch)
    with psycopg.connect(admin_dsn, autocommit=True) as connection:
        connection.execute(SQL("CREATE DATABASE {}").format(Identifier(scratch)))
    try:
        apply_sql(dsn)
        yield dsn
    finally:
        with psycopg.connect(admin_dsn, autocommit=True) as connection:
            connection.execute(SQL("DROP DATABASE {} WITH (FORCE)").format(Identifier(scratch)))


@pytest.fixture
def harness(monkeypatch: pytest.MonkeyPatch, postgres_dsn: str) -> Iterator[Harness]:
    connection = psycopg.connect(postgres_dsn, autocommit=True, row_factory=dict_row)
    connection.execute("TRUNCATE benchmark_runs, evaluations CASCADE")
    connection.execute("INSERT INTO users(id) VALUES (%s), (%s) ON CONFLICT DO NOTHING", (USER_A, USER_B))
    for offset, (evaluation_id, owner, visibility) in enumerate(
        (
            ("ev_a_private", USER_A, "private"),
            ("ev_b_private", USER_B, "private"),
            ("ev_a_public", USER_A, "public"),
            ("ev_a_team", USER_A, "team"),
            ("ev_a_org", USER_A, "org"),
        )
    ):
        connection.execute(
            "INSERT INTO evaluations(id, owner_id, name, task_id, task_revision, visibility, created_at) "
            "VALUES (%s, %s, %s, 'task_synthetic', 1, %s, '2026-01-01'::timestamptz + %s * interval '1 second')",
            (evaluation_id, owner, evaluation_id, visibility, offset),
        )
    store = FakeEntityStore()
    monkeypatch.setattr(entity_store, "entity_client", lambda: store)
    monkeypatch.setattr(entity_store, "private_workspace", lambda: "scaled-evals-internal")
    monkeypatch.setattr(auth, "platform_auth_enabled", lambda: True)
    monkeypatch.setattr(auth, "current_auth_context", AUTH_CONTEXT.get)
    monkeypatch.setattr(settings._instance, "entity_store_reads_enabled", False)
    seed_task_revision("task_synthetic", owner_id=USER_A, image_ref="synthetic.invalid/task:fixture")
    app = FastAPI()
    app.include_router(stamp_router_authz(evaluations.router), prefix="/apis/scaled-evals/v1")
    app.include_router(stamp_router_authz(benchmark_runs.router), prefix="/apis/scaled-evals/v1")
    admin_probes: list[str] = []

    @app.middleware("http")
    async def synthetic_admission(
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        user = request.headers.get("x-repro-user")
        if user not in (USER_A, USER_B):
            return JSONResponse(status_code=401, content={"detail": "synthetic principal required"})
        token = AUTH_CONTEXT.set(AuthContext(principal_id=user, principal_on_behalf_of=request.headers.get("x-act-as")))
        try:
            return await call_next(request)
        finally:
            AUTH_CONTEXT.reset(token)

    def connection_dependency() -> Iterator[psycopg.Connection[DictRow]]:
        yield connection

    def synthetic_admin(request: Request) -> bool:
        admin_probes.append(request.url.path)
        return request.headers.get("x-repro-admin") == "true"

    @contextmanager
    def stream_database() -> Iterator[Database]:
        yield Database(connection)

    def unreadable_secret(reference: str) -> bool:
        raise AssertionError("unauthorized evaluation inspected credential references")

    app.dependency_overrides[get_stream_database_factory] = lambda: stream_database
    app.dependency_overrides[caller_secret_readable] = lambda: unreadable_secret
    app.dependency_overrides[get_conn] = connection_dependency
    app.dependency_overrides[caller_is_admin] = synthetic_admin
    try:
        with TestClient(app) as client:
            yield Harness(client, connection, admin_probes)
    finally:
        app.dependency_overrides.clear()
        connection.close()


@pytest.mark.parametrize("operation", ["get", "cancel", "delete"])
@pytest.mark.parametrize(
    "owner,other,evaluation_id",
    [
        (USER_A, USER_B, "ev_a_private"),
        (USER_B, USER_A, "ev_b_private"),
    ],
)
def test_regression_private_by_id_denies_non_owner(
    harness: Harness,
    operation: Operation,
    owner: str,
    other: str,
    evaluation_id: str,
) -> None:
    assert auth.current_auth_context() is None
    before = harness.row(evaluation_id)
    assert before.owner_id == owner
    response = harness.request(operation, evaluation_id, other)
    after = harness.row(evaluation_id)
    assert (response.status_code in (403, 404), after == before) == (True, True)


@pytest.mark.parametrize("user,hidden_id", [(USER_A, "ev_b_private"), (USER_B, "ev_a_private")])
@pytest.mark.parametrize("query", ["", "?mine=false", "?q=private", "?limit=1&order=asc"])
def test_regression_default_list_hides_other_private_rows(
    harness: Harness,
    user: str,
    hidden_id: str,
    query: str,
) -> None:
    response = harness.client.get(BASE + query, headers={"x-repro-user": user})
    assert response.status_code == 200
    body = ListEnvelope[Evaluation].model_validate(response.json())
    seen = list(body.data)
    while body.next_cursor:
        separator = "&" if query else "?"
        response = harness.client.get(
            BASE + query + separator + "cursor=" + body.next_cursor, headers={"x-repro-user": user}
        )
        assert response.status_code == 200
        body = ListEnvelope[Evaluation].model_validate(response.json())
        seen.extend(body.data)
        assert len(seen) <= 5
    print("LIST", json.dumps({"user": user, "query": query, "ids": [row.id for row in seen]}))
    assert hidden_id not in [row.id for row in seen]


@pytest.mark.parametrize("operation", ["get", "cancel", "delete"])
@pytest.mark.parametrize("admin", [False, True])
def test_owner_and_explicit_admin_controls(harness: Harness, operation: Operation, admin: bool) -> None:
    response = harness.request(operation, "ev_a_private", USER_B if admin else USER_A, admin=admin)
    assert response.status_code == 200
    if operation == "get":
        assert EvaluationResponse.model_validate(response.json()).id == "ev_a_private"
        assert harness.row("ev_a_private").owner_id == USER_A
    elif operation == "cancel":
        assert harness.row("ev_a_private").status == "cancelled"
    else:
        assert harness.row("ev_a_private").deleted_at is not None
    assert harness.admin_probes


@pytest.mark.parametrize("user,owned_id", [(USER_A, "ev_a_private"), (USER_B, "ev_b_private")])
def test_mine_and_shared_filters(harness: Harness, user: str, owned_id: str) -> None:
    for query, expected in (
        ("?mine=true", {owned_id} | ({"ev_a_public", "ev_a_team", "ev_a_org"} if user == USER_A else set())),
        ("?shared=true", {"ev_a_public", "ev_a_team", "ev_a_org"}),
        ("?mine=true&shared=true", {"ev_a_public", "ev_a_team", "ev_a_org"} if user == USER_A else set()),
    ):
        response = harness.client.get(BASE + query, headers={"x-repro-user": user})
        assert response.status_code == 200
        ids = {row.id for row in ListEnvelope[Evaluation].model_validate(response.json()).data}
        print("FILTER", user, query, sorted(ids))
        assert ids == expected


def test_public_read_control(harness: Harness) -> None:
    assert harness.request("get", "ev_a_public", USER_B).status_code == 200


@pytest.mark.parametrize("operation", ["cancel", "delete"])
def test_proposed_shared_mutation_requires_owner(harness: Harness, operation: Operation) -> None:
    before = harness.row("ev_a_public")
    response = harness.request(operation, "ev_a_public", USER_B)
    assert (response.status_code in (403, 404), harness.row("ev_a_public") == before) == (True, True)


@pytest.mark.parametrize("operation", ["get", "cancel", "delete"])
def test_unknown_and_deleted_controls(harness: Harness, operation: Operation) -> None:
    assert harness.request(operation, "ev_missing", USER_B).status_code == 404
    harness.connection.execute("UPDATE evaluations SET deleted_at = NOW() WHERE id = 'ev_a_private'")
    assert harness.request(operation, "ev_a_private", USER_A).status_code == 404
    response = harness.client.get(BASE, headers={"x-repro-user": USER_B})
    assert "ev_a_private" not in [row.id for row in ListEnvelope[Evaluation].model_validate(response.json()).data]


def test_synthetic_anonymous_admission_control(harness: Harness) -> None:
    assert harness.client.get(BASE).status_code == 401


def test_task_store_denial_does_not_mask_private_isolation(harness: Harness, monkeypatch: pytest.MonkeyPatch) -> None:
    def denied_revisions(self: TaskStore, pairs: Iterable[tuple[str, int]]) -> Never:
        raise HTTPException(status_code=403, detail="synthetic task-store denial")

    monkeypatch.setattr(TaskStore, "revisions", denied_revisions)
    assert harness.request("get", "ev_a_private", USER_A).status_code == 403
    assert harness.request("get", "ev_a_private", USER_B).status_code == 404
    before = harness.row("ev_a_private")
    assert harness.request("cancel", "ev_a_private", USER_B).status_code == 404
    assert harness.request("delete", "ev_a_private", USER_B).status_code == 404
    assert harness.row("ev_a_private") == before


def test_cancel_is_idempotent_control(harness: Harness) -> None:
    assert harness.request("cancel", "ev_a_private", USER_A).status_code == 200
    before = harness.row("ev_a_private")
    assert harness.request("cancel", "ev_a_private", USER_A).status_code == 200
    assert harness.row("ev_a_private") == before
    count = harness.connection.execute(
        "SELECT count(*) AS count FROM evaluation_events WHERE evaluation_id = 'ev_a_private' AND status = 'cancelled'"
    ).fetchone()
    assert count is not None and count["count"] == 2


def test_synthetic_admin_list_control(harness: Harness) -> None:
    response = harness.client.get(BASE, headers={"x-repro-user": USER_B, "x-repro-admin": "true"})
    ids = {row.id for row in ListEnvelope[Evaluation].model_validate(response.json()).data}
    assert {"ev_a_private", "ev_b_private", "ev_a_public"} <= ids
    assert harness.admin_probes


@pytest.mark.parametrize(
    "path",
    [
        "telemetry",
        "reproduce",
        "logs",
        "logs/stream",
        "events",
        "events/stream",
        "artifacts",
        "artifacts/file.txt",
        "archive",
        "archive/download",
        "harbor-viewer/archive",
    ],
)
def test_private_auxiliary_routes_deny_before_loading(harness: Harness, path: str) -> None:
    response = harness.client.get(f"{BASE}/ev_a_private/{path}", headers={"x-repro-user": USER_B})
    assert response.status_code == 404


@pytest.mark.parametrize("path", ["retry", "archive"])
@pytest.mark.parametrize("visibility", ["private", "public", "team", "org"])
def test_readable_rows_do_not_authorize_mutations(harness: Harness, path: str, visibility: str) -> None:
    harness.connection.execute(
        "UPDATE evaluations SET visibility = %s, status = 'failed', "
        "archive_status = 'ready', credentials = %s WHERE id = 'ev_a_private'",
        (visibility, psycopg.types.json.Json({"openai": "owner/private-key"})),
    )
    before = harness.row("ev_a_private")
    response = harness.client.post(
        f"{BASE}/ev_a_private/{path}", json={"force": True}, headers={"x-repro-user": USER_B}
    )
    assert response.status_code == 404
    assert harness.row("ev_a_private") == before


def test_terminal_cancel_fallback_remains_scoped(harness: Harness) -> None:
    harness.connection.execute("UPDATE evaluations SET status = 'succeeded' WHERE id = 'ev_a_private'")
    assert harness.request("cancel", "ev_a_private", USER_B).status_code == 404
    assert harness.request("cancel", "ev_a_private", USER_A).status_code == 200


def test_private_null_owner_is_admin_only_and_development_remains_accessible(
    harness: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness.connection.execute("UPDATE evaluations SET owner_id = NULL WHERE id = 'ev_a_private'")
    assert harness.request("get", "ev_a_private", USER_A).status_code == 404
    assert harness.request("get", "ev_a_private", USER_B, admin=True).status_code == 200
    monkeypatch.setattr(auth, "platform_auth_enabled", lambda: False)
    assert harness.request("get", "ev_a_private", USER_A).status_code == 200


def test_delegated_owner_uses_effective_principal(harness: Harness) -> None:
    response = harness.client.get(f"{BASE}/ev_a_private", headers={"x-repro-user": USER_B, "x-act-as": USER_A})
    assert response.status_code == 200


def test_admin_mine_is_still_a_narrowing_filter(harness: Harness) -> None:
    response = harness.client.get(f"{BASE}?mine=true", headers={"x-repro-user": USER_B, "x-repro-admin": "true"})
    assert response.status_code == 200
    assert {row["id"] for row in response.json()["data"]} == {"ev_b_private"}


def test_projection_cannot_resurrect_hidden_or_deleted_rows(harness: Harness, monkeypatch: pytest.MonkeyPatch) -> None:
    from nemo_scaled_evals_plugin import projection

    def forbidden_reader() -> projection.EvaluationProjectionReader:
        raise AssertionError("projection was consulted before SQL eligibility")

    monkeypatch.setattr(settings._instance, "entity_store_reads_enabled", True)
    monkeypatch.setattr(projection, "evaluation_reader", forbidden_reader)
    assert harness.request("get", "ev_a_private", USER_B).status_code == 404
    harness.connection.execute("UPDATE evaluations SET deleted_at = NOW() WHERE id = 'ev_a_private'")
    assert harness.request("get", "ev_a_private", USER_A).status_code == 404


def seed_run(harness: Harness, *, visibility: str = "private", terminal: bool = False) -> None:
    harness.connection.execute(
        "INSERT INTO benchmark_runs(id, owner_id, name, benchmark_id, benchmark_revision, visibility) "
        "VALUES ('run_owner', %s, 'owner run', 'benchmark_fixture', 1, %s)",
        (USER_A, visibility),
    )
    harness.connection.execute(
        "UPDATE evaluations SET benchmark_run_id = 'run_owner', visibility = %s WHERE id = 'ev_a_private'",
        (visibility,),
    )
    if terminal:
        harness.connection.execute(
            "UPDATE evaluations SET status = 'succeeded', evidence_status = 'ready', archive_status = 'ready' "
            "WHERE benchmark_run_id = 'run_owner'"
        )


@pytest.mark.parametrize("suffix", ["", "/reproduce", "/evaluations", "/archive", "/archive/download"])
def test_benchmark_read_routes_do_not_bypass_isolation(harness: Harness, suffix: str) -> None:
    seed_run(harness)
    response = harness.client.get(
        f"/apis/scaled-evals/v1/benchmark-runs/run_owner{suffix}", headers={"x-repro-user": USER_B}
    )
    assert response.status_code == 404


@pytest.mark.parametrize("operation", ["cancel", "delete", "archive"])
def test_benchmark_mutation_does_not_touch_other_owners(harness: Harness, operation: str) -> None:
    seed_run(harness, visibility="public")
    before = harness.row("ev_a_private")
    path = "/apis/scaled-evals/v1/benchmark-runs/run_owner"
    if operation == "delete":
        response = harness.client.delete(path, headers={"x-repro-user": USER_B})
    else:
        response = harness.client.post(f"{path}/{operation}", json={}, headers={"x-repro-user": USER_B})
    assert response.status_code == 404
    assert harness.row("ev_a_private") == before
    row = harness.connection.execute(
        "SELECT cancelled_at, deleted_at FROM benchmark_runs WHERE id = 'run_owner'"
    ).fetchone()
    assert row is not None and row["cancelled_at"] is None and row["deleted_at"] is None


def test_benchmark_delete_atomically_soft_deletes_members(harness: Harness) -> None:
    seed_run(harness, terminal=True)
    response = harness.client.delete("/apis/scaled-evals/v1/benchmark-runs/run_owner", headers={"x-repro-user": USER_A})
    assert response.status_code == 200
    assert harness.row("ev_a_private").deleted_at is not None
    assert harness.request("get", "ev_a_private", USER_A).status_code == 404
    row = harness.connection.execute("SELECT deleted_at FROM benchmark_runs WHERE id = 'run_owner'").fetchone()
    assert row is not None and row["deleted_at"] is not None


@pytest.mark.parametrize("pending", ["active", "cleanup", "evidence", "archive"])
def test_benchmark_delete_conflict_preserves_parent_and_members(harness: Harness, pending: str) -> None:
    seed_run(harness, terminal=True)
    updates = {
        "active": "status = 'running'",
        "cleanup": "cancel_teardown_status = 'pending'",
        "evidence": "evidence_status = 'building'",
        "archive": "archive_status = 'building'",
    }
    harness.connection.execute(f"UPDATE evaluations SET {updates[pending]} WHERE id = 'ev_a_private'")
    before = harness.row("ev_a_private")
    response = harness.client.delete("/apis/scaled-evals/v1/benchmark-runs/run_owner", headers={"x-repro-user": USER_A})
    assert response.status_code == 409
    assert harness.row("ev_a_private") == before
    row = harness.connection.execute("SELECT deleted_at FROM benchmark_runs WHERE id = 'run_owner'").fetchone()
    assert row is not None and row["deleted_at"] is None


def test_mixed_owner_bulk_cancel_has_no_partial_writes(harness: Harness) -> None:
    seed_run(harness)
    harness.connection.execute("UPDATE evaluations SET benchmark_run_id = 'run_owner' WHERE id = 'ev_b_private'")
    before = harness.row("ev_a_private")
    response = harness.client.post(
        "/apis/scaled-evals/v1/benchmark-runs/run_owner/cancel", headers={"x-repro-user": USER_A}
    )
    assert response.status_code == 404
    assert harness.row("ev_a_private") == before
    row = harness.connection.execute("SELECT cancelled_at FROM benchmark_runs WHERE id = 'run_owner'").fetchone()
    assert row is not None and row["cancelled_at"] is None


@pytest.mark.parametrize("query", ["", "?q=owner", "?limit=1&order=asc"])
def test_benchmark_list_enforces_scope_before_search_and_paging(harness: Harness, query: str) -> None:
    seed_run(harness)
    path = f"/apis/scaled-evals/v1/benchmark-runs{query}"
    assert harness.client.get(path, headers={"x-repro-user": USER_B}).json()["data"] == []
    response = harness.client.get(path, headers={"x-repro-user": USER_A})
    assert response.status_code == 200
    assert [row["id"] for row in response.json()["data"]] == ["run_owner"]


def test_owner_can_cancel_a_run_and_its_members(harness: Harness) -> None:
    seed_run(harness)
    response = harness.client.post(
        "/apis/scaled-evals/v1/benchmark-runs/run_owner/cancel", headers={"x-repro-user": USER_A}
    )
    assert response.status_code == 200
    assert harness.row("ev_a_private").status == "cancelled"


def test_event_polling_queries_recheck_access(harness: Harness) -> None:
    harness.connection.execute(
        "INSERT INTO evaluation_events(evaluation_id, status, detail) VALUES ('ev_a_private', 'queued', 'owner event')"
    )
    owner = EvaluationRepository(harness.connection, access=ResourceAccess(USER_A, False))
    other = EvaluationRepository(harness.connection, access=ResourceAccess(USER_B, False))
    assert owner.list_events("ev_a_private", limit=10, cursor=None, offset=0)
    assert owner.load_event_batch("ev_a_private", limit=10)
    assert other.list_events("ev_a_private", limit=10, cursor=None, offset=0) == []
    assert other.load_event_batch("ev_a_private", limit=10) == []
    harness.connection.execute("UPDATE evaluations SET deleted_at = NOW() WHERE id = 'ev_a_private'")
    assert owner.load_event_batch("ev_a_private", limit=10) == []
    assert owner.load_stream_status("ev_a_private") is None


def test_public_parent_does_not_expose_private_foreign_members(harness: Harness) -> None:
    seed_run(harness, visibility="public")
    harness.connection.execute("UPDATE evaluations SET benchmark_run_id = 'run_owner' WHERE id = 'ev_b_private'")
    response = harness.client.get("/apis/scaled-evals/v1/benchmark-runs/run_owner", headers={"x-repro-user": USER_A})
    assert response.status_code == 404


@pytest.mark.parametrize("operation", ["delete", "archive"])
def test_mixed_owner_bulk_operations_do_not_partially_change_state(harness: Harness, operation: str) -> None:
    seed_run(harness, terminal=True)
    harness.connection.execute("UPDATE evaluations SET benchmark_run_id = 'run_owner' WHERE id = 'ev_b_private'")
    before = harness.row("ev_a_private")
    path = "/apis/scaled-evals/v1/benchmark-runs/run_owner"
    if operation == "delete":
        response = harness.client.delete(path, headers={"x-repro-user": USER_A})
    else:
        response = harness.client.post(path + "/archive", json={}, headers={"x-repro-user": USER_A})
    assert response.status_code == 404
    assert harness.row("ev_a_private") == before


def test_owner_can_retry_without_inspecting_another_owners_secrets(harness: Harness) -> None:
    harness.connection.execute("UPDATE evaluations SET status = 'failed' WHERE id = 'ev_a_private'")
    response = harness.client.post(f"{BASE}/ev_a_private/retry", headers={"x-repro-user": USER_A})
    assert response.status_code == 202
    assert harness.row("ev_a_private").status == "queued"


def test_owner_can_queue_and_read_a_benchmark_archive(harness: Harness) -> None:
    seed_run(harness, terminal=True)
    harness.connection.execute(
        "UPDATE evaluations SET archive_object_key = 'fixture/results.tar.gz', "
        "archive_built_at = NOW(), archive_size_bytes = 123 WHERE id = 'ev_a_private'"
    )
    path = "/apis/scaled-evals/v1/benchmark-runs/run_owner/archive"
    response = harness.client.post(path, json={}, headers={"x-repro-user": USER_A})
    assert response.status_code == 202
    assert response.json()["members"][0]["id"] == "ev_a_private"
    assert harness.client.get(path, headers={"x-repro-user": USER_A}).status_code == 200
    assert harness.client.get(path, headers={"x-repro-user": USER_B}).status_code == 404


def test_cascade_retains_artifacts_and_previously_deleted_members(harness: Harness) -> None:
    seed_run(harness, terminal=True)
    harness.connection.execute(
        "UPDATE evaluations SET archive_object_key = 'retained/results.tar.gz' WHERE id = 'ev_a_private'"
    )
    harness.connection.execute(
        "UPDATE evaluations SET benchmark_run_id = 'run_owner', deleted_at = NOW() WHERE id = 'ev_a_public'"
    )
    before = harness.row("ev_a_public")
    assert BenchmarkRunRepository(harness.connection, access=ResourceAccess(USER_A, False)).soft_delete("run_owner")
    assert harness.row("ev_a_public") == before
    row = harness.connection.execute(
        "SELECT archive_object_key, deleted_at FROM evaluations WHERE id = 'ev_a_private'"
    ).fetchone()
    assert row is not None and row["archive_object_key"] == "retained/results.tar.gz"
    assert row["deleted_at"] is not None


def test_cascade_rolls_back_member_updates_if_parent_update_fails(harness: Harness) -> None:
    seed_run(harness, terminal=True)
    harness.connection.execute(
        "ALTER TABLE benchmark_runs ADD CONSTRAINT owner_test_no_delete CHECK (deleted_at IS NULL)"
    )
    try:
        repository = BenchmarkRunRepository(harness.connection, access=ResourceAccess(USER_A, False))
        with pytest.raises(psycopg.errors.CheckViolation):
            repository.soft_delete("run_owner")
        assert harness.row("ev_a_private").deleted_at is None
    finally:
        harness.connection.execute("ALTER TABLE benchmark_runs DROP CONSTRAINT owner_test_no_delete")


def test_cascade_serializes_with_concurrent_member_retry(harness: Harness, postgres_dsn: str) -> None:
    seed_run(harness, terminal=True)
    harness.connection.execute("UPDATE evaluations SET status = 'failed' WHERE id = 'ev_a_private'")
    started = Event()

    def retry_member() -> DictRow | None:
        with psycopg.connect(postgres_dsn, autocommit=True, row_factory=dict_row) as connection:
            started.set()
            return EvaluationRepository(connection, access=ResourceAccess(USER_A, False)).retry_failed("ev_a_private")

    with ThreadPoolExecutor(max_workers=1) as executor:
        with harness.connection.transaction():
            harness.connection.execute("SELECT id FROM benchmark_runs WHERE id = 'run_owner' FOR UPDATE")
            pending = executor.submit(retry_member)
            assert started.wait(timeout=5)
            assert not pending.done()
            repository = BenchmarkRunRepository(harness.connection, access=ResourceAccess(USER_A, False))
            assert repository.soft_delete("run_owner")
        assert pending.result(timeout=10) is None
    assert harness.row("ev_a_private").deleted_at is not None
    assert harness.row("ev_a_private").status == "failed"


def test_retry_winning_before_delete_prevents_cascade(harness: Harness) -> None:
    seed_run(harness, terminal=True)
    harness.connection.execute("UPDATE evaluations SET status = 'failed' WHERE id = 'ev_a_private'")
    access = ResourceAccess(USER_A, False)
    assert EvaluationRepository(harness.connection, access=access).retry_failed("ev_a_private") is not None
    with pytest.raises(Conflict):
        BenchmarkRunRepository(harness.connection, access=access).soft_delete("run_owner")
    assert harness.row("ev_a_private").deleted_at is None
    assert harness.row("ev_a_private").status == "queued"
