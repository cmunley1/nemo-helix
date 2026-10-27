# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The trial-credential routes record, end and look up which trial each credential belongs to."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from nemo_evals.api.v2 import trial_credentials
from nemo_helix_plugin.authz import PATH_RULES_ATTR, CallerKind
from nemo_helix_plugin.entity_client import get_entity_client

DIGEST = "cred-" + "a" * 58
STARTED = datetime(2026, 10, 26, 12, tzinfo=UTC)
BASE = "/v2/workspaces/models/trial-credentials"


def _body(**overrides: Any) -> dict[str, Any]:
    return {
        "name": DIGEST,
        "session_id": "run-1:hello__abc",
        "trace_id": "0af7651916cd43dd8448eb211c80319c",
        "root_span_id": "b7ad6b7169203331",
        "intake_workspace": "evals",
        "started_at": STARTED.isoformat(),
        "evaluation_name": "run-1",
        "test_case_name": "hello",
        **overrides,
    }


@pytest.fixture
def client(entity_store) -> TestClient:
    app = FastAPI()
    app.include_router(trial_credentials.router, prefix="/v2/workspaces/{workspace}")
    app.dependency_overrides[get_entity_client] = lambda: entity_store
    return TestClient(app)


def test_a_recorded_credential_is_looked_up_by_its_name(client: TestClient) -> None:
    created = client.post(BASE, json=_body())
    assert created.status_code == 200, created.text

    found = client.get(f"{BASE}/{DIGEST}")
    assert found.status_code == 200
    assert found.json() | {"started_at": None} == created.json() | {"started_at": None}
    assert found.json()["workspace"] == "models"
    assert found.json()["ended_at"] is None


def test_recording_the_same_binding_again_returns_it(client: TestClient) -> None:
    assert client.post(BASE, json=_body()).status_code == 200
    again = client.post(BASE, json=_body())
    assert again.status_code == 200
    assert again.json()["session_id"] == "run-1:hello__abc"


def test_a_credential_cannot_be_rebound_to_another_trial(client: TestClient) -> None:
    assert client.post(BASE, json=_body()).status_code == 200
    assert client.post(BASE, json=_body(session_id="run-1:other__def")).status_code == 409


def test_a_credential_no_trial_holds_is_not_found(client: TestClient) -> None:
    assert client.get(f"{BASE}/{'cred-' + 'b' * 58}").status_code == 404
    assert client.patch(f"{BASE}/{'cred-' + 'b' * 58}", json={"ended_at": STARTED.isoformat()}).status_code == 404


@pytest.mark.parametrize("name", ["not-a-name", "cred-" + "A" * 58, "a" * 64])
def test_a_credential_is_named_from_its_lowercase_sha256(client: TestClient, name: str) -> None:
    assert client.post(BASE, json=_body(name=name)).status_code == 422
    assert client.get(f"{BASE}/{name}").status_code == 422


def test_ending_a_credential_records_when_and_is_idempotent(client: TestClient) -> None:
    assert client.post(BASE, json=_body()).status_code == 200
    ended_at = (STARTED + timedelta(minutes=5)).isoformat()

    ended = client.patch(f"{BASE}/{DIGEST}", json={"ended_at": ended_at})
    assert ended.status_code == 200
    assert datetime.fromisoformat(ended.json()["ended_at"]) == STARTED + timedelta(minutes=5)
    assert client.patch(f"{BASE}/{DIGEST}", json={"ended_at": ended_at}).status_code == 200
    assert datetime.fromisoformat(client.get(f"{BASE}/{DIGEST}").json()["ended_at"]) == STARTED + timedelta(minutes=5)


def test_a_credential_ends_once(client: TestClient) -> None:
    assert client.post(BASE, json=_body()).status_code == 200
    assert client.patch(f"{BASE}/{DIGEST}", json={"ended_at": (STARTED + timedelta(minutes=5)).isoformat()})
    later = client.patch(f"{BASE}/{DIGEST}", json={"ended_at": (STARTED + timedelta(minutes=9)).isoformat()})
    assert later.status_code == 409


def test_a_credential_cannot_end_before_it_was_issued(client: TestClient) -> None:
    assert client.post(BASE, json=_body()).status_code == 200
    early = client.patch(f"{BASE}/{DIGEST}", json={"ended_at": (STARTED - timedelta(seconds=1)).isoformat()})
    assert early.status_code == 422


def test_only_services_may_use_the_routes() -> None:
    """The gateway's middleware and evaluation jobs are services; no user reads or writes these records."""
    for route in trial_credentials.router.routes:
        rules = getattr(route, "endpoint").__dict__[PATH_RULES_ATTR]
        assert [rule.callers for rule in rules] == [[CallerKind.SERVICE_PRINCIPAL]]


def test_the_routes_are_left_out_of_the_api_schema() -> None:
    """Only the evaluation job and the trial-recording middleware use these records; they aren't public API."""
    app = FastAPI()
    app.include_router(trial_credentials.router, prefix="/v2/workspaces/{workspace}")

    assert not [path for path in app.openapi()["paths"] if "trial-credentials" in path]
