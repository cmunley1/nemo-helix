# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Each Harbor trial calls the gateway with its own trial token, which names the trial and ends with it."""

from __future__ import annotations

import contextlib
import json
from collections.abc import Generator
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
    issuing,
    with_trial_credentials,
)
from nemo_evals.intake.mapping import session_id_for
from nemo_evals.jobs.gateway_agents import GATEWAY_PLACEHOLDER_KEY
from nemo_evals.trial_tokens import (
    TOKEN_LIFETIME_SEC,
    ModelAccess,
    TrialClaims,
    TrialPolicy,
    TrialTokenReader,
    unsigned_trial_token,
)
from nemo_helix_plugin.client.client import AsyncNemoClient
from nhx_evals_sdk.agent_eval.runtimes.harbor.runtime import HarborRuntimeConfig
from nhx_evals_sdk.agent_eval.runtimes.provenance import require_no_plaintext_credentials
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest
from opentelemetry.proto.trace.v1.trace_pb2 import Span

TRIAL = "hello-world__AbC123"
RUN = "run-1-task-0"
KEY_ENV = "OPENAI_API_KEY"
CLAIMS = TrialClaims(
    session=session_id_for(RUN, TRIAL),
    attempt=0,
    intake_workspace="evals",
    evaluation="run-1",
    policy=TrialPolicy(model_access=ModelAccess.CLOSED),
)


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


ACCESS_KEYS = "/apis/auth/v2/access-keys"
OTLP = "/apis/intake/v2/workspaces/evals/ingest/otlp/v1/traces"


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
        intake_workspace="evals",
        evaluation_name="run-1",
        model_access=ModelAccess.CLOSED,
    )


def _trial_dir(tmp_path: Path, name: str = TRIAL) -> Path:
    trial_dir = tmp_path / name
    trial_dir.mkdir(exist_ok=True)
    return trial_dir


@pytest.mark.parametrize("agent_key", [GATEWAY_PLACEHOLDER_KEY, None])
async def test_without_a_user_key_a_trial_gets_an_unsigned_trial_token(
    platform: Platform, tmp_path: Path, agent_key: str | None
) -> None:
    """Platform auth is off whenever the agent holds the placeholder, so there is no key to sign a token."""
    issued = await _issuer().issue(_trial_dir(tmp_path), agent_key=agent_key)

    assert issued is not None and issued.access_key_jti is None
    assert issued.trial == CLAIMS
    assert await TrialTokenReader(None).trial(issued.credential) == CLAIMS
    assert platform.seen == []


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


async def test_with_a_user_key_a_trial_gets_an_access_key_carrying_its_claims(
    platform: Platform, tmp_path: Path
) -> None:
    platform.answers[("POST", ACCESS_KEYS)] = _minted()

    issued = await _issuer().issue(_trial_dir(tmp_path), agent_key="users-own-key")

    assert issued == IssuedCredential("minted-token", CLAIMS, "ak_" + "1" * 32)
    [key_request] = [json.loads(content) for content in platform.bodies("POST", ACCESS_KEYS)]
    assert key_request["scope"] == ["inference"]
    assert key_request["name"] == f"nemo-evals-trial {CLAIMS.session}"
    assert key_request["expires_in_seconds"] == TOKEN_LIFETIME_SEC
    assert key_request["service_attributes"] == CLAIMS.service_attributes()


async def test_each_attempt_at_a_trial_is_counted(platform: Platform, tmp_path: Path) -> None:
    """Harbor retries a trial under the same name, so its attempts share a session and differ by attempt."""
    issuer = _issuer()

    attempts = [await issuer.issue(_trial_dir(tmp_path), agent_key=None) for _ in range(2)]
    other = await issuer.issue(_trial_dir(tmp_path, "hello-world__ZyX987"), agent_key=None)

    assert [issued.trial.attempt for issued in attempts if issued is not None] == [0, 1]
    assert other is not None and other.trial.attempt == 0


async def test_a_key_that_cannot_be_minted_leaves_the_agent_its_users_key(platform: Platform, tmp_path: Path) -> None:
    platform.answers[("POST", ACCESS_KEYS)] = httpx.Response(404, json={"detail": "Scoped Access Keys are not enabled"})

    assert await _issuer().issue(_trial_dir(tmp_path), agent_key="users-own-key") is None


def _root_spans(platform: Platform) -> list[Span]:
    return [
        span
        for export in platform.bodies("POST", OTLP)
        for resource in ExportTraceServiceRequest.FromString(export).resource_spans
        for scope in resource.scope_spans
        for span in scope.spans
    ]


async def test_ending_a_token_records_the_trials_root_then_revokes_its_key(platform: Platform) -> None:
    jti = "ak_" + "3" * 32
    platform.answers[("DELETE", f"{ACCESS_KEYS}/{jti}")] = httpx.Response(200, json={"jti": jti, "revoked": True})
    platform.answers[("POST", OTLP)] = httpx.Response(200, json={})

    await _issuer().end(IssuedCredential("minted-token", CLAIMS, jti))

    paths = [(request.method, request.url.path) for request in platform.seen]
    assert paths == [("POST", OTLP), ("DELETE", f"{ACCESS_KEYS}/{jti}")]
    [span] = _root_spans(platform)
    assert span.trace_id.hex() == CLAIMS.trace_id and span.span_id.hex() == CLAIMS.root_span_id
    assert not span.parent_span_id


async def test_the_root_span_runs_from_the_first_attempt_to_the_last(platform: Platform, tmp_path: Path) -> None:
    """Every attempt rewrites the same root span, so each must give it the trial's first start."""
    platform.answers[("POST", OTLP)] = httpx.Response(200, json={})
    issuer = _issuer()

    for _ in range(2):
        issued = await issuer.issue(_trial_dir(tmp_path), agent_key=None)
        assert issued is not None
        await issuer.end(issued)

    first, last = _root_spans(platform)
    assert first.span_id == last.span_id
    assert first.start_time_unix_nano == last.start_time_unix_nano
    assert last.end_time_unix_nano > first.end_time_unix_nano


async def test_an_unreachable_platform_never_fails_the_trial(platform: Platform, tmp_path: Path) -> None:
    assert await _issuer().issue(_trial_dir(tmp_path), agent_key="users-own-key") is None
    await _issuer().end(IssuedCredential("token", CLAIMS, "ak_" + "5" * 32))


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
    issued = IssuedCredential(unsigned_trial_token(CLAIMS), CLAIMS, None)
    issuer = FakeIssuer(issued)
    agent, environment = _agent(tmp_path), Environment()

    with issuing(issuer):  # ty: ignore[invalid-argument-type]  # a stand-in issuer
        await agent.setup(environment)
        await agent.run("do it", environment, AgentContext())

    assert issuer.issued_for == [(TRIAL, GATEWAY_PLACEHOLDER_KEY)]
    assert agent.saw_in_setup == agent.saw_in_run == issued.credential
    assert environment.overlays == [{KEY_ENV: issued.credential}]
    assert issuer.ended == [issued]


class CrashingAgent(RecordingAgent):
    async def run(self, instruction: str, environment: Any, context: AgentContext) -> None:
        raise RuntimeError("agent crashed")


async def test_an_agent_that_fails_ends_its_credential(tmp_path: Path) -> None:
    issued = IssuedCredential(unsigned_trial_token(CLAIMS), CLAIMS, None)
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


def test_the_wrapped_config_passes_the_runtimes_credential_check() -> None:
    """The runtime refuses a run whose agent_kwargs look like plaintext credentials, so the wrapper's must not."""
    config = with_trial_credentials(_config(agent_name="codex"), KEY_ENV)

    require_no_plaintext_credentials(config.agent_kwargs, field="agent_kwargs", alternative="env_secrets")


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
