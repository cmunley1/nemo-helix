# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A trial token names its trial in claims that only evals can set, and readers trust no other issuer."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import Any

import pytest
from nemo_evals.intake.mapping import session_id_for
from nemo_evals.trial_tokens import (
    REJECTED_TOKEN_CACHE_SEC,
    TOKEN_LIFETIME_SEC,
    ModelAccess,
    TrialClaims,
    TrialPolicy,
    TrialTokenReader,
    claims_a_trial,
    unsigned_token_claims,
    unsigned_trial_token,
)
from nemo_helix_plugin.auth.access_keys.types import AccessKeyCreateRequest
from nemo_helix_plugin.client.oidc import decode_jwt_claims, decode_jwt_header, generate_unsigned_jwt

TRIAL = TrialClaims(
    session=session_id_for("run-1", "hello__abc1234"),
    attempt=0,
    intake_workspace="evals",
    evaluation="bench",
    task="trial-rec/hello-world",
    policy=TrialPolicy(model_access=ModelAccess.CLOSED),
)


class Clock:
    def __init__(self, now: float) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


class Verifier:
    """Trusts the claims of the tokens it was given, and counts what it was asked."""

    def __init__(self, *tokens: str) -> None:
        self._tokens = set(tokens)
        self.asked = 0

    async def __call__(self, token: str) -> Mapping[str, Any] | None:
        self.asked += 1
        return decode_jwt_claims(token) if token in self._tokens else None


def _claims(**evals: Any) -> dict[str, Any]:
    return {"nhx_service_attributes": {"evals": {**TRIAL.service_attributes(), **evals}}}


def test_the_trace_and_root_span_are_the_ones_publishing_derives_from_the_session() -> None:
    digest = hashlib.sha256(TRIAL.session.encode()).hexdigest()

    assert (TRIAL.trace_id, TRIAL.root_span_id) == (digest[:32], digest[32:48])


def test_the_claims_fit_what_the_auth_service_accepts_as_service_attributes() -> None:
    request = AccessKeyCreateRequest(name="trial", service_attributes=TRIAL.service_attributes())

    assert request.service_attributes == {
        "v": 1,
        "kind": "trial",
        "session": "run-1:hello__abc1234",
        "attempt": 0,
        "intake_workspace": "evals",
        "evaluation": "bench",
        "task": "trial-rec/hello-world",
        "policy": {"model_access": "closed"},
    }


@pytest.mark.parametrize(
    ("claims", "trial"),
    [
        pytest.param(_claims(), TRIAL, id="a trial"),
        pytest.param(_claims(v=2), None, id="a shape this reader doesn't know"),
        pytest.param(_claims(kind="judge"), None, id="another evals use"),
        pytest.param(_claims(policy={"model_access": "anything"}), None, id="an unknown policy"),
        pytest.param({"nhx_service_attributes": {"other": TRIAL.service_attributes()}}, None, id="another service"),
        pytest.param({"evals": TRIAL.service_attributes()}, None, id="outside the service attributes"),
        pytest.param({}, None, id="no claims"),
    ],
)
def test_only_the_evals_attributes_of_a_trial_name_one(claims: dict[str, Any], trial: TrialClaims | None) -> None:
    assert TrialClaims.from_token_claims(claims) == trial


def test_an_unsigned_trial_token_comes_from_nemo_evals_and_lives_a_day() -> None:
    token = unsigned_trial_token(TRIAL)
    claims = decode_jwt_claims(token)

    assert decode_jwt_header(token)["alg"] == "none"
    assert claims["iss"] == "nemo-evals"
    assert claims["exp"] - claims["iat"] == TOKEN_LIFETIME_SEC
    assert TrialClaims.from_token_claims(claims) == TRIAL
    assert claims_a_trial(token)


@pytest.mark.parametrize("token", ["users-own-key", "a.b.c", generate_unsigned_jwt("alice@example.com")])
def test_a_token_that_names_no_trial_doesnt_claim_one(token: str) -> None:
    assert not claims_a_trial(token)


def _unsigned(issuer: str = "nemo-evals", expires_in_seconds: int = 60, alg: str = "none") -> str:
    token = generate_unsigned_jwt(
        "nemo-evals", issuer=issuer, expires_in_seconds=expires_in_seconds, extra_claims=_claims(), issued_at=1000
    )
    if alg == "none":
        return token
    _, payload, _ = token.split(".")
    return f"eyJhbGciOiJSUzI1NiJ9.{payload}.c2lnbmF0dXJl"


@pytest.mark.parametrize(
    ("token", "trusted"),
    [
        pytest.param(_unsigned(), True, id="from nemo-evals"),
        pytest.param(_unsigned(issuer="someone-else"), False, id="another issuer"),
        pytest.param(_unsigned(expires_in_seconds=-1), False, id="expired"),
        pytest.param(_unsigned(alg="RS256"), False, id="claims a signature it isn't checked for"),
    ],
)
def test_with_auth_off_only_unexpired_unsigned_tokens_from_nemo_evals_are_trusted(token: str, trusted: bool) -> None:
    assert (unsigned_token_claims(token, now=1030) is not None) is trusted


async def test_with_auth_off_a_token_is_skipped_once_it_expires() -> None:
    clock = Clock(1030)
    reader = TrialTokenReader(None, clock=clock)
    token = _unsigned()

    assert await reader.trial(token) == TRIAL
    clock.now = 1061
    assert await reader.trial(token) is None


async def test_with_auth_on_an_unsigned_token_is_never_trusted() -> None:
    """``verify`` decides alone: an unsigned token from nemo-evals counts for nothing when a verifier is set."""
    reader = TrialTokenReader(Verifier(), clock=Clock(1030))

    assert await reader.trial(_unsigned()) is None


async def test_a_token_is_checked_once_until_it_expires() -> None:
    token = unsigned_trial_token(TRIAL)
    verify, clock = Verifier(token), Clock(0)
    reader = TrialTokenReader(verify, clock=clock)

    assert await reader.trial(token) == TRIAL
    assert await reader.trial(token) == TRIAL
    assert verify.asked == 1

    clock.now = decode_jwt_claims(token)["exp"]
    await reader.trial(token)
    assert verify.asked == 2


async def test_a_rejected_token_is_checked_again_after_a_minute() -> None:
    verify, clock = Verifier(), Clock(0)
    reader = TrialTokenReader(verify, clock=clock)

    assert await reader.trial("a.b.c") is None
    assert await reader.trial("a.b.c") is None
    clock.now = REJECTED_TOKEN_CACHE_SEC
    assert await reader.trial("a.b.c") is None

    assert verify.asked == 2


async def test_a_failed_check_is_not_remembered() -> None:
    attempts = 0

    async def unreachable(token: str) -> Mapping[str, Any] | None:
        nonlocal attempts
        attempts += 1
        raise RuntimeError("signing keys unreachable")

    reader = TrialTokenReader(unreachable)
    for _ in range(2):
        with pytest.raises(RuntimeError):
            await reader.trial("a.b.c")

    assert attempts == 2


async def test_the_reader_remembers_at_most_its_limit_of_tokens() -> None:
    tokens = [unsigned_trial_token(TRIAL.model_copy(update={"attempt": n})) for n in range(3)]
    verify = Verifier(*tokens)
    reader = TrialTokenReader(verify, max_cached_tokens=2)

    for token in tokens:
        await reader.trial(token)
    await reader.trial(tokens[0])

    assert len(reader._entries) == 2
    assert verify.asked == 4
