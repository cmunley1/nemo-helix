# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Each Harbor trial calls the gateway with its own credential, recorded as the trial's and ended with it."""

from __future__ import annotations

import contextlib
import hashlib
import json
from collections.abc import Generator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import pytest
from harbor.agents.nop import NopAgent
from harbor.models.agent.context import AgentContext
from nemo_evals.harbor import trial_credentials
from nemo_evals.harbor.agent_setup_egress import INNER_AGENT_KWARG as SETUP_INNER_KWARG
from nemo_evals.harbor.agent_setup_egress import SETUP_EGRESS_AGENT, SETUP_HOSTS_KWARG
from nemo_evals.harbor.trial_credentials import (
    CREDENTIAL_ENV_KWARG,
    INNER_AGENT_KWARG,
    TRIAL_CREDENTIAL_AGENT,
    IssuedCredential,
    TrialCredentialIssuer,
    TrialIdentity,
    issuing,
    with_trial_credentials,
)
from nemo_evals.intake.mapping import session_id_for
from nemo_evals.jobs.gateway_agents import GATEWAY_PLACEHOLDER_KEY
from nemo_evals.trial_recording import TRIAL_ACCESS_KEY_NAME_PREFIX, is_trial_credential
from nemo_helix_plugin.client.client import AsyncNemoClient
from nhx_evals_sdk.agent_eval.runtimes.harbor.runtime import HarborRuntimeConfig
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest

TRIAL = "hello-world__AbC123"
RUN = "run-1-task-0"
KEY_ENV = "OPENAI_API_KEY"


def _digest(credential: str) -> str:
    return hashlib.sha256(credential.encode()).hexdigest()


def test_a_trials_identity_is_the_session_publishing_uses_and_is_stable() -> None:
    identity = TrialIdentity.of(RUN, TRIAL)

    assert identity.session_id == session_id_for(RUN, TRIAL)
    assert identity == TrialIdentity.of(RUN, TRIAL)
    assert len(identity.trace_id) == 32 and len(identity.root_span_id) == 16
    assert TrialIdentity.of(RUN, "other__x").trace_id != identity.trace_id


class Platform:
    """Answers the issuer's platform calls by method and path, and keeps every request it saw."""

    def __init__(self, answers: dict[tuple[str, str], httpx.Response]) -> None:
        self.answers = answers
        self.seen: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.seen.append(request)
        return self.answers.get((request.method, request.url.path), httpx.Response(500, json={"detail": "unexpected"}))

    def bodies(self, method: str, path: str) -> list[Any]:
        return [request.content for request in self.seen if request.method == method and request.url.path == path]


CREDENTIALS = "/apis/evals/v2/workspaces/models/trial-credentials"
ACCESS_KEYS = "/apis/auth/v2/access-keys"
OTLP = "/apis/intake/v2/workspaces/evals/ingest/otlp/v1/traces"


def _trial_json(digest: str, **overrides: Any) -> dict[str, Any]:
    identity = TrialIdentity.of(RUN, TRIAL)
    return {
        "name": digest,
        "workspace": "models",
        "session_id": identity.session_id,
        "trace_id": identity.trace_id,
        "root_span_id": identity.root_span_id,
        "intake_workspace": "evals",
        "started_at": datetime(2026, 10, 26, 12, tzinfo=UTC).isoformat(),
        **overrides,
    }


@pytest.fixture
def platform(monkeypatch: pytest.MonkeyPatch) -> Platform:
    platform = Platform({})
    real_client = httpx.AsyncClient

    def client(**kwargs: Any) -> httpx.AsyncClient:
        return real_client(transport=httpx.MockTransport(platform), **kwargs)

    monkeypatch.setattr(trial_credentials.httpx, "AsyncClient", client)
    return platform


def _issuer() -> TrialCredentialIssuer:
    return TrialCredentialIssuer(
        AsyncNemoClient(base_url="http://platform.test"),
        run_id=RUN,
        credential_workspace="models",
        intake_workspace="evals",
        evaluation_name="run-1",
    )


def _trial_dir(tmp_path: Path) -> Path:
    trial_dir = tmp_path / TRIAL
    trial_dir.mkdir()
    return trial_dir


@pytest.mark.parametrize("agent_key", [GATEWAY_PLACEHOLDER_KEY, None])
async def test_without_a_user_key_a_trial_gets_a_random_token(
    platform: Platform, tmp_path: Path, agent_key: str | None
) -> None:
    """Platform auth is off whenever the agent holds the placeholder: the gateway only needs a unique value."""
    platform.answers[("POST", CREDENTIALS)] = httpx.Response(200, json=_trial_json("a" * 64))

    issued = await _issuer().issue(_trial_dir(tmp_path), agent_key=agent_key)

    assert issued is not None and issued.access_key_jti is None
    assert issued.credential not in {GATEWAY_PLACEHOLDER_KEY, None}
    assert is_trial_credential(issued.credential)
    [body] = [json.loads(content) for content in platform.bodies("POST", CREDENTIALS)]
    assert body["name"] == _digest(issued.credential)
    assert body["session_id"] == session_id_for(RUN, TRIAL)
    assert body["intake_workspace"] == "evals" and body["evaluation_name"] == "run-1"
    assert platform.bodies("POST", ACCESS_KEYS) == []


def _minted(token: str = "minted-token", jti: str = "ak_" + "1" * 32) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "jti": jti,
            "principal": "user@example.com",
            "status": "ACTIVE",
            "issuer": "nemo",
            "audiences": ["nemo"],
            "scope": ["inference"],
            "created_at": "2026-10-26T12:00:00Z",
            "token": token,
            "token_type": "Bearer",
        },
    )


async def test_with_a_user_key_a_trial_gets_its_own_inference_scoped_access_key(
    platform: Platform, tmp_path: Path
) -> None:
    platform.answers[("POST", ACCESS_KEYS)] = _minted()
    platform.answers[("POST", CREDENTIALS)] = httpx.Response(200, json=_trial_json(_digest("minted-token")))

    issued = await _issuer().issue(_trial_dir(tmp_path), agent_key="users-own-key")

    assert issued == IssuedCredential("minted-token", _digest("minted-token"), "ak_" + "1" * 32)
    [key_request] = [json.loads(content) for content in platform.bodies("POST", ACCESS_KEYS)]
    assert key_request["scope"] == ["inference"]
    assert key_request["name"].startswith(TRIAL_ACCESS_KEY_NAME_PREFIX)
    assert key_request["expires_in_seconds"] > 0
    [body] = [json.loads(content) for content in platform.bodies("POST", CREDENTIALS)]
    assert body["name"] == _digest("minted-token")


async def test_a_key_that_cannot_be_minted_leaves_the_agent_its_users_key(platform: Platform, tmp_path: Path) -> None:
    platform.answers[("POST", ACCESS_KEYS)] = httpx.Response(404, json={"detail": "Scoped Access Keys are not enabled"})

    assert await _issuer().issue(_trial_dir(tmp_path), agent_key="users-own-key") is None
    assert platform.bodies("POST", CREDENTIALS) == []


async def test_a_minted_key_that_cannot_be_recorded_is_revoked(platform: Platform, tmp_path: Path) -> None:
    jti = "ak_" + "2" * 32
    platform.answers[("POST", ACCESS_KEYS)] = _minted(jti=jti)
    platform.answers[("POST", CREDENTIALS)] = httpx.Response(403, json={"detail": "Forbidden"})
    platform.answers[("DELETE", f"{ACCESS_KEYS}/{jti}")] = httpx.Response(200, json={"jti": jti, "revoked": True})

    assert await _issuer().issue(_trial_dir(tmp_path), agent_key="users-own-key") is None
    assert len(platform.bodies("DELETE", f"{ACCESS_KEYS}/{jti}")) == 1


async def test_ending_a_credential_records_the_trials_root_then_revokes_its_key(platform: Platform) -> None:
    jti = "ak_" + "3" * 32
    digest = _digest("minted-token")
    platform.answers[("DELETE", f"{ACCESS_KEYS}/{jti}")] = httpx.Response(200, json={"jti": jti, "revoked": True})
    platform.answers[("PATCH", f"{CREDENTIALS}/{digest}")] = httpx.Response(
        200, json=_trial_json(digest, ended_at="2026-10-26T12:05:00Z", test_case_name="hello")
    )
    platform.answers[("POST", OTLP)] = httpx.Response(200, json={})

    await _issuer().end(IssuedCredential("minted-token", digest, jti))

    paths = [(request.method, request.url.path) for request in platform.seen]
    assert paths == [("PATCH", f"{CREDENTIALS}/{digest}"), ("POST", OTLP), ("DELETE", f"{ACCESS_KEYS}/{jti}")]
    [export] = platform.bodies("POST", OTLP)
    request = ExportTraceServiceRequest.FromString(export)
    [span] = request.resource_spans[0].scope_spans[0].spans
    identity = TrialIdentity.of(RUN, TRIAL)
    assert span.trace_id.hex() == identity.trace_id and span.span_id.hex() == identity.root_span_id
    assert not span.parent_span_id
    assert span.end_time_unix_nano - span.start_time_unix_nano == 5 * 60 * 1_000_000_000


async def test_an_unreachable_platform_never_fails_the_trial(platform: Platform, tmp_path: Path) -> None:
    assert await _issuer().issue(_trial_dir(tmp_path), agent_key=None) is None
    await _issuer().end(IssuedCredential("token", _digest("token"), None))


# --- the agent wrapper ----------------------------------------------------------------------------------


class Environment:
    def __init__(self) -> None:
        self.overlays: list[dict[str, str]] = []

    @contextlib.contextmanager
    def scoped_exec_env(self, env: dict[str, str]) -> Generator[None]:
        self.overlays.append(env)
        yield


class RecordingAgent(NopAgent):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.saw_in_setup: str | None = None
        self.saw_in_run: str | None = None

    async def setup(self, environment: Any) -> None:
        self.saw_in_setup = self._extra_env.get(KEY_ENV)

    async def run(self, instruction: str, environment: Any, context: AgentContext) -> None:
        self.saw_in_run = self.extra_env.get(KEY_ENV)


class FakeIssuer:
    def __init__(self, issued: IssuedCredential | None) -> None:
        self.issued = issued
        self.issued_for: list[tuple[str, str | None]] = []
        self.ended: list[IssuedCredential] = []

    async def issue(self, trial_dir: Path, *, agent_key: str | None) -> IssuedCredential | None:
        self.issued_for.append((trial_dir.name, agent_key))
        return self.issued

    async def end(self, issued: IssuedCredential) -> None:
        self.ended.append(issued)


def _agent(tmp_path: Path) -> Any:
    wrapped = trial_credentials._with_trial_credential(RecordingAgent, KEY_ENV)
    return wrapped(logs_dir=tmp_path / TRIAL / "agent", extra_env={KEY_ENV: GATEWAY_PLACEHOLDER_KEY})


async def test_the_agent_calls_with_its_trials_credential_from_setup_on(tmp_path: Path) -> None:
    issued = IssuedCredential("trial-token", _digest("trial-token"), None)
    issuer = FakeIssuer(issued)
    agent, environment = _agent(tmp_path), Environment()

    with issuing(issuer):  # ty: ignore[invalid-argument-type]  # a stand-in issuer
        await agent.setup(environment)
        await agent.run("do it", environment, AgentContext())

    assert issuer.issued_for == [(TRIAL, GATEWAY_PLACEHOLDER_KEY)]
    assert agent.saw_in_setup == agent.saw_in_run == "trial-token"
    assert environment.overlays == [{KEY_ENV: "trial-token"}]
    assert issuer.ended == [issued]


class CrashingAgent(RecordingAgent):
    async def run(self, instruction: str, environment: Any, context: AgentContext) -> None:
        raise RuntimeError("agent crashed")


async def test_an_agent_that_fails_ends_its_credential(tmp_path: Path) -> None:
    issued = IssuedCredential("trial-token", _digest("trial-token"), None)
    issuer = FakeIssuer(issued)
    wrapped = trial_credentials._with_trial_credential(CrashingAgent, KEY_ENV)
    agent: Any = wrapped(logs_dir=tmp_path / TRIAL / "agent", extra_env={KEY_ENV: GATEWAY_PLACEHOLDER_KEY})

    with issuing(issuer):  # ty: ignore[invalid-argument-type]  # a stand-in issuer
        await agent.setup(Environment())
        with pytest.raises(RuntimeError, match="agent crashed"):
            await agent.run("do it", Environment(), AgentContext())

    assert issuer.ended == [issued]


async def test_without_an_issued_credential_the_agent_keeps_its_key(tmp_path: Path) -> None:
    issuer = FakeIssuer(None)
    agent, environment = _agent(tmp_path), Environment()

    with issuing(issuer):  # ty: ignore[invalid-argument-type]  # a stand-in issuer
        await agent.setup(environment)
        await agent.run("do it", environment, AgentContext())

    assert agent.saw_in_run == GATEWAY_PLACEHOLDER_KEY
    assert environment.overlays == [] and issuer.ended == []


def _config(**overrides: Any) -> HarborRuntimeConfig:
    return HarborRuntimeConfig(jobs_dir=Path("/tmp/jobs"), **overrides)


def test_the_config_wraps_the_agent_and_names_its_key_variable() -> None:
    config = with_trial_credentials(_config(agent_name="claude-code", agent_kwargs={"x": 1}), KEY_ENV)

    assert config.agent_import_path == TRIAL_CREDENTIAL_AGENT
    assert config.agent_kwargs == {"x": 1, INNER_AGENT_KWARG: "claude-code", CREDENTIAL_ENV_KWARG: KEY_ENV}
    assert config.agent_setup_timeout_multiplier is not None


def test_the_config_wraps_an_agent_already_wrapped_for_setup_egress() -> None:
    egress = _config(
        agent_name="claude-code",
        agent_import_path=SETUP_EGRESS_AGENT,
        agent_kwargs={SETUP_INNER_KWARG: "claude-code", SETUP_HOSTS_KWARG: ["registry.npmjs.org"]},
        agent_setup_timeout_multiplier=2.5,
    )

    config = with_trial_credentials(egress, KEY_ENV)

    assert config.agent_kwargs[INNER_AGENT_KWARG] == SETUP_EGRESS_AGENT
    assert config.agent_setup_timeout_multiplier == 2.5
    kwargs = dict(config.agent_kwargs)
    kwargs.pop(INNER_AGENT_KWARG)
    agent_class = trial_credentials._inner_agent_class(SETUP_EGRESS_AGENT, kwargs)
    assert agent_class.name() == "claude-code"
    assert SETUP_INNER_KWARG not in kwargs and SETUP_HOSTS_KWARG not in kwargs


@pytest.mark.parametrize(
    "config",
    [
        pytest.param(_config(agent_name="oracle"), id="oracle"),
        pytest.param(_config(agent_name="nop"), id="nop"),
        pytest.param(
            _config(agent_import_path="my_agent:Agent", agent_dir=Path("/agents/mine")), id="loose agent file"
        ),
    ],
)
def test_agents_that_cannot_or_need_not_be_wrapped_are_left_alone(config: HarborRuntimeConfig) -> None:
    assert with_trial_credentials(config, KEY_ENV) == config
