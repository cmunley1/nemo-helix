# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Which trial each issued gateway credential belongs to: the records, and a client for their service-only routes.

The evaluation job writes these records and the trial-recording middleware reads them. The routes are left out of
the public API schema, so this module, not the public SDK, holds their types and client.
"""

from __future__ import annotations

from abc import abstractmethod
from datetime import datetime

from nemo_helix_plugin.client.client import AsyncNemoClient
from nemo_helix_plugin.client.endpoint import get, patch, post
from nemo_helix_plugin.client.method import method
from pydantic import AwareDatetime, BaseModel, ConfigDict, Field

_CREDENTIAL_RECORD_NAME = r"^cred-[0-9a-f]{58}$"
_TRACE_ID_HEX = r"^[0-9a-f]{32}$"
_SPAN_ID_HEX = r"^[0-9a-f]{16}$"


class _TrialCredentialFields(BaseModel):
    session_id: str = Field(min_length=1, max_length=1024, description="The trial's Intake session.")
    trace_id: str = Field(pattern=_TRACE_ID_HEX, description="The trial's trace id: 32 lowercase hex characters.")
    root_span_id: str = Field(
        pattern=_SPAN_ID_HEX, description="The trial's root span id: 16 lowercase hex characters."
    )
    intake_workspace: str = Field(min_length=1, description="The workspace the trial's spans are recorded in.")
    started_at: AwareDatetime = Field(description="When the trial was issued the credential.")
    evaluation_name: str | None = Field(default=None, description="The Intake evaluation the trial belongs to.")
    test_case_name: str | None = Field(default=None, description="The Intake test case the trial runs.")


class CreateTrialCredentialRequest(_TrialCredentialFields):
    """Bind a credential, by its SHA-256, to the trial it was issued to."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(
        pattern=_CREDENTIAL_RECORD_NAME,
        description="`cred-` and the first 58 hex digits of the credential's SHA-256, which fits an entity name.",
    )


class EndTrialCredentialRequest(BaseModel):
    """Stop attributing calls made with the credential after ``ended_at``."""

    model_config = ConfigDict(extra="forbid")

    ended_at: AwareDatetime


class TrialCredential(_TrialCredentialFields):
    """The trial a credential was issued to; the Inference Gateway attributes calls made with it to that trial."""

    name: str = Field(
        description="`cred-` and the first 58 hex digits of the credential's SHA-256, which fits an entity name."
    )
    workspace: str
    ended_at: AwareDatetime | None = Field(default=None, description="When the trial stopped using the credential.")

    def covers(self, at: datetime) -> bool:
        """Whether a call made at ``at`` belongs to the trial."""
        return self.started_at <= at and (self.ended_at is None or at <= self.ended_at)


_TRIAL_CREDENTIALS = "/apis/evals/v2/workspaces/{workspace}/trial-credentials"


@post(_TRIAL_CREDENTIALS)
@abstractmethod
def create_trial_credential(*, workspace: str | None = None, body: CreateTrialCredentialRequest) -> TrialCredential: ...


@patch(f"{_TRIAL_CREDENTIALS}/{{name}}")
@abstractmethod
def end_trial_credential(
    *, workspace: str | None = None, name: str, body: EndTrialCredentialRequest
) -> TrialCredential: ...


@get(f"{_TRIAL_CREDENTIALS}/{{name}}")
@abstractmethod
def get_trial_credential(*, workspace: str | None = None, name: str) -> TrialCredential: ...


class AsyncTrialCredentialsClient(AsyncNemoClient):
    """Async client for the trial-credential routes."""

    create_trial_credential = method(create_trial_credential)
    end_trial_credential = method(end_trial_credential)
    get_trial_credential = method(get_trial_credential)
