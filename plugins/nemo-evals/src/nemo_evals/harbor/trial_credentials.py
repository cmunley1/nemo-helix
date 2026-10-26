# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Give each Harbor trial its own credential for the Inference Gateway, so the gateway can attribute its calls.

Harbor builds the agent from :data:`TRIAL_CREDENTIAL_AGENT` with the real agent in the ``trial_credential_inner``
kwarg and the variable that holds the agent's gateway key in ``trial_credential_env``. What it gets back is an
instance of a subclass of the real agent that, before setup, asks the job's :class:`TrialCredentialIssuer` for a
credential and puts it in that variable, and ends the credential when the agent finishes.

With platform auth off, the gateway ignores credentials, so the agent's placeholder key is replaced with a random
token. With auth on, the agent holds a key the user supplied; it is replaced with a Scoped Access Key minted for
the trial on the user's behalf, scoped to inference, and revoked when the trial ends. If that key can't be minted,
the agent keeps the user's key and its calls are not attributed.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import logging
import secrets
from collections.abc import AsyncGenerator, Generator, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import cache
from pathlib import Path
from typing import Any, ClassVar

import httpx
from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext
from harbor.models.task.task import Task
from harbor.models.trial.config import TrialConfig
from nemo_evals.harbor.agent_setup_egress import (
    SETUP_EGRESS_AGENT,
    UNWRAPPED_AGENTS,
    AgentSetupEgress,
    inner_agent_class,
    setup_egress_agent_class,
    setup_timeout_multiplier,
)
from nemo_evals.intake.mapping import session_id_for
from nemo_evals.jobs.gateway_agents import GATEWAY_PLACEHOLDER_KEY
from nemo_evals.trial_spans import export_request, trial_root_span
from nemo_helix_plugin.auth.access_keys.client import AsyncAccessKeysClient
from nemo_helix_plugin.auth.access_keys.types import AccessKeyCreateRequest
from nemo_helix_plugin.client.client import AsyncNemoClient
from nemo_helix_plugin.evals.client import AsyncEvaluatorClient
from nemo_helix_plugin.evals.types import CreateTrialCredentialRequest, EndTrialCredentialRequest
from nemo_helix_plugin.intake.client import AsyncIntakeClient
from nhx_evals_sdk.agent_eval.runtimes.harbor.runtime import HarborRuntimeConfig

logger = logging.getLogger(__name__)

TRIAL_CREDENTIAL_AGENT = "nemo_evals.harbor.trial_credentials:TrialCredentialAgent"
INNER_AGENT_KWARG = "trial_credential_inner"
CREDENTIAL_ENV_KWARG = "trial_credential_env"

REQUEST_TIMEOUT_SEC = 10.0
ACCESS_KEY_SCOPE = "inference"
ACCESS_KEY_LIFETIME_SEC = 24 * 60 * 60


@dataclass(frozen=True)
class TrialIdentity:
    """The Intake session, trace and root span every producer of one trial's spans shares."""

    session_id: str
    trace_id: str
    root_span_id: str

    @classmethod
    def of(cls, run_id: str, trial_name: str) -> TrialIdentity:
        """The identity of trial ``trial_name`` in SDK run ``run_id``: the session publishing already uses."""
        session_id = session_id_for(run_id, trial_name)
        digest = hashlib.sha256(session_id.encode()).hexdigest()
        return cls(session_id=session_id, trace_id=digest[:32], root_span_id=digest[32:48])


@dataclass(frozen=True)
class IssuedCredential:
    credential: str
    digest: str
    access_key_jti: str | None


class TrialCredentialIssuer:
    """Issues trials their gateway credentials and records them in the gateway model's workspace."""

    def __init__(
        self,
        client: AsyncNemoClient,
        *,
        run_id: str,
        credential_workspace: str,
        intake_workspace: str,
        evaluation_name: str | None = None,
    ) -> None:
        self._client = client
        self._run_id = run_id
        self._credential_workspace = credential_workspace
        self._intake_workspace = intake_workspace
        self._evaluation_name = evaluation_name

    @contextlib.asynccontextmanager
    async def _platform(self) -> AsyncGenerator[AsyncNemoClient]:
        # Harbor runs trials on its own event loop, so each call gets a transport bound to the current one.
        event_hooks = {event: list(hooks) for event, hooks in self._client._client.event_hooks.items()}
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT_SEC, event_hooks=event_hooks) as http_client:
            yield self._client.with_http_client(http_client)

    async def issue(self, trial_dir: Path, *, agent_key: str | None) -> IssuedCredential | None:
        """A credential for the trial in ``trial_dir``, recorded as the trial's; ``None`` to keep ``agent_key``."""
        identity = TrialIdentity.of(self._run_id, trial_dir.name)
        issued: IssuedCredential | None = None
        try:
            async with asyncio.timeout(REQUEST_TIMEOUT_SEC * 2), self._platform() as client:
                if agent_key is None or agent_key == GATEWAY_PLACEHOLDER_KEY:
                    issued = _random_credential()
                else:
                    issued = await _access_key(client, identity.session_id)
                body = CreateTrialCredentialRequest(
                    name=issued.digest,
                    session_id=identity.session_id,
                    trace_id=identity.trace_id,
                    root_span_id=identity.root_span_id,
                    intake_workspace=self._intake_workspace,
                    started_at=datetime.now(UTC),
                    evaluation_name=self._evaluation_name,
                    test_case_name=_test_case_name(trial_dir),
                )
                evals = AsyncEvaluatorClient.from_client(client)
                (await evals.create_trial_credential(workspace=self._credential_workspace, body=body)).data()
        except Exception:
            logger.warning("Could not issue trial %s its own gateway credential", identity.session_id, exc_info=True)
            if issued is not None and issued.access_key_jti is not None:
                await self._revoke(issued.access_key_jti)
            return None
        return issued

    async def _revoke(self, jti: str) -> None:
        try:
            async with asyncio.timeout(REQUEST_TIMEOUT_SEC), self._platform() as client:
                await AsyncAccessKeysClient.from_client(client).revoke_access_key(jti=jti)
        except Exception:
            logger.warning("Could not revoke a trial's access key %s", jti, exc_info=True)

    async def end(self, issued: IssuedCredential) -> None:
        """End the credential now: record the trial's root span, then revoke a minted key."""
        try:
            async with asyncio.timeout(REQUEST_TIMEOUT_SEC * 2), self._platform() as client:
                evals = AsyncEvaluatorClient.from_client(client)
                trial = (
                    await evals.end_trial_credential(
                        workspace=self._credential_workspace,
                        name=issued.digest,
                        body=EndTrialCredentialRequest(ended_at=datetime.now(UTC)),
                    )
                ).data()
                await AsyncIntakeClient.from_client(client).create_otlp_traces(
                    workspace=trial.intake_workspace, content=export_request([trial_root_span(trial)])
                )
        except Exception:
            logger.warning("Could not end a trial's gateway credential", exc_info=True)
        if issued.access_key_jti is not None:
            await self._revoke(issued.access_key_jti)


def _random_credential() -> IssuedCredential:
    credential = secrets.token_urlsafe(32)
    return IssuedCredential(credential, hashlib.sha256(credential.encode()).hexdigest(), None)


async def _access_key(client: AsyncNemoClient, session_id: str) -> IssuedCredential:
    request = AccessKeyCreateRequest(
        name=f"trial {session_id}"[:128],
        description="Inference Gateway key for one evaluation trial; revoked when the trial ends.",
        expires_in_seconds=ACCESS_KEY_LIFETIME_SEC,
        scope=[ACCESS_KEY_SCOPE],
    )
    key = (await AsyncAccessKeysClient.from_client(client).create_access_key(body=request)).data()
    return IssuedCredential(key.token, hashlib.sha256(key.token.encode()).hexdigest(), key.jti)


def _test_case_name(trial_dir: Path) -> str | None:
    try:
        task_dir = TrialConfig.model_validate_json((trial_dir / "config.json").read_text()).task.path
        return Task(task_dir).name if task_dir is not None else None
    except Exception:
        logger.debug("Could not read the task of trial %s", trial_dir.name, exc_info=True)
        return None


_ISSUER: TrialCredentialIssuer | None = None


@contextlib.contextmanager
def issuing(issuer: TrialCredentialIssuer) -> Generator[None]:
    """Make ``issuer`` the one this process's trials get their credentials from, for the block's duration."""
    global _ISSUER
    previous, _ISSUER = _ISSUER, issuer
    try:
        yield
    finally:
        _ISSUER = previous


def current_issuer() -> TrialCredentialIssuer | None:
    """The issuer :func:`issuing` installed, or ``None`` when trials keep the agent's key."""
    return _ISSUER


@cache
def _with_trial_credential(agent_class: type[BaseAgent], env_name: str) -> type[BaseAgent]:
    class TrialCredentialAgent(agent_class):  # ty: ignore[unsupported-base]  # a class known only at run time
        _trial_credential: IssuedCredential | None = None

        async def setup(self, environment: BaseEnvironment) -> None:
            issuer = current_issuer()
            if issuer is not None:
                self._trial_credential = await issuer.issue(
                    Path(self.logs_dir).parent, agent_key=self._extra_env.get(env_name)
                )
            if self._trial_credential is None:
                await super().setup(environment)
                return
            self._extra_env[env_name] = self._trial_credential.credential
            try:
                with environment.scoped_exec_env({env_name: self._trial_credential.credential}):
                    await super().setup(environment)
            except BaseException:
                await self._end_trial_credential()
                raise

        async def run(self, instruction: str, environment: BaseEnvironment, context: AgentContext) -> None:
            try:
                await super().run(instruction, environment, context)
            finally:
                await self._end_trial_credential()

        async def _end_trial_credential(self) -> None:
            issued, self._trial_credential = self._trial_credential, None
            issuer = current_issuer()
            if issued is not None and issuer is not None:
                await asyncio.shield(issuer.end(issued))

    TrialCredentialAgent.__name__ = agent_class.__name__
    TrialCredentialAgent.__qualname__ = agent_class.__qualname__
    TrialCredentialAgent.__module__ = agent_class.__module__
    return TrialCredentialAgent


def _inner_agent_class(inner: str, kwargs: dict[str, Any]) -> type[BaseAgent]:
    return setup_egress_agent_class(kwargs) if inner == SETUP_EGRESS_AGENT else inner_agent_class(inner, kwargs)


class TrialCredentialAgent:
    """Stands in for the real agent class in Harbor's ``AgentConfig.import_path``; never instantiated itself."""

    DEFAULT_SETUP_TIMEOUT_SEC: ClassVar[float] = 360.0

    def __new__(cls, *args: Any, **kwargs: Any) -> BaseAgent:
        inner = kwargs.pop(INNER_AGENT_KWARG)
        env_name = kwargs.pop(CREDENTIAL_ENV_KWARG)
        return _with_trial_credential(_inner_agent_class(inner, kwargs), env_name)(*args, **kwargs)

    @classmethod
    def preflight(cls, kwargs: Mapping[str, Any] | None = None, env: Mapping[str, str] | None = None) -> None:
        """The real agent's preflight, without the wrapper's kwargs."""
        options = dict(kwargs or {})
        inner = options.pop(INNER_AGENT_KWARG)
        options.pop(CREDENTIAL_ENV_KWARG, None)
        if inner == SETUP_EGRESS_AGENT:
            AgentSetupEgress.preflight(kwargs=options, env=env)
        else:
            inner_agent_class(inner, options).preflight(kwargs=options, env=env)


def with_trial_credentials(config: HarborRuntimeConfig, api_key_env: str) -> HarborRuntimeConfig:
    """``config`` with its agent wrapped so each trial gets its own credential in ``api_key_env``.

    An agent from a loose file (``agent_dir``) can't be wrapped, and ``oracle`` and ``nop`` call no model.
    """
    agent = config.agent_import_path or config.agent_name or "oracle"
    if config.agent_dir is not None or agent in UNWRAPPED_AGENTS:
        return config
    update: dict[str, Any] = {
        "agent_import_path": TRIAL_CREDENTIAL_AGENT,
        "agent_kwargs": {**config.agent_kwargs, INNER_AGENT_KWARG: agent, CREDENTIAL_ENV_KWARG: api_key_env},
    }
    if agent != SETUP_EGRESS_AGENT:
        # Harbor falls back to the global multiplier, then 1.0, when the setup one is unset.
        multiplier = config.agent_setup_timeout_multiplier
        if multiplier is None:
            multiplier = config.timeout_multiplier if config.timeout_multiplier is not None else 1.0
        update["agent_setup_timeout_multiplier"] = setup_timeout_multiplier(agent, multiplier)
    return config.model_copy(update=update)
