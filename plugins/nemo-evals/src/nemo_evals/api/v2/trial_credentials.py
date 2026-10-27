# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Which trial each issued credential belongs to, under /apis/evals/v2/workspaces/{workspace}/trial-credentials.

Service-only, and left out of the public API schema: the types and client live in
:mod:`nemo_evals.trial_credential_records`.

An evaluation job issues each trial its own credential for the Inference Gateway and records it here by a
name derived from its SHA-256. The gateway's trial-recording middleware looks the caller's credential up to
attribute its calls.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Path, status
from nemo_evals.authz import scope
from nemo_evals.entities import TrialCredentialEntity
from nemo_evals.trial_credential_records import CreateTrialCredentialRequest, EndTrialCredentialRequest, TrialCredential
from nemo_helix_plugin.authz import CallerKind, PermissionSet, path_rule, perm
from nemo_helix_plugin.entity_client import (
    NemoEntitiesClient,
    NemoEntityConflictError,
    NemoEntityNotFoundError,
    get_entity_client,
)


class TrialCredentialPerms(PermissionSet, namespace="evaluator.trial-credentials"):
    """Permissions for the trial credential records services keep for each other."""

    CREATE = perm("Record which trial a credential was issued to")
    READ = perm("Look up which trial a credential was issued to")
    UPDATE = perm("Record when a trial stopped using its credential")


CredentialName = Annotated[
    str, Path(pattern=r"^cred-[0-9a-f]{58}$", description="The credential's record name, from its SHA-256.")
]

router = APIRouter()


def _to_wire(entity: TrialCredentialEntity) -> TrialCredential:
    return TrialCredential.model_validate(entity.model_dump(include=set(TrialCredential.model_fields)))


async def _get(entities: NemoEntitiesClient, workspace: str, name: str) -> TrialCredentialEntity:
    try:
        return await entities.get(TrialCredentialEntity, name=name, workspace=workspace)
    except NemoEntityNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No trial holds that credential") from e


@router.post(
    "/trial-credentials",
    summary="Record Trial Credential",
    include_in_schema=False,
    responses={status.HTTP_409_CONFLICT: {"description": "The credential is recorded for another trial"}},
)
@scope.write
@path_rule(callers=[CallerKind.SERVICE_PRINCIPAL], permissions=[TrialCredentialPerms.CREATE])
async def create_trial_credential(
    workspace: str,
    body: CreateTrialCredentialRequest,
    entities: NemoEntitiesClient = Depends(get_entity_client),
) -> TrialCredential:
    """Record the trial a credential was issued to. Recording the same binding again returns it."""
    entity = TrialCredentialEntity(workspace=workspace, **body.model_dump())
    try:
        return _to_wire(await entities.create(entity))
    except NemoEntityConflictError:
        existing = await _get(entities, workspace, body.name)
    if existing.model_dump(include=set(CreateTrialCredentialRequest.model_fields)) != body.model_dump():
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="The credential is recorded for another trial")
    return _to_wire(existing)


@router.patch(
    "/trial-credentials/{name}",
    summary="End Trial Credential",
    include_in_schema=False,
    responses={
        status.HTTP_404_NOT_FOUND: {"description": "No trial holds the credential"},
        status.HTTP_409_CONFLICT: {"description": "The credential already ended at another time"},
        status.HTTP_422_UNPROCESSABLE_CONTENT: {"description": "The end is before the trial was issued it"},
    },
)
@scope.write
@path_rule(callers=[CallerKind.SERVICE_PRINCIPAL], permissions=[TrialCredentialPerms.UPDATE])
async def end_trial_credential(
    workspace: str,
    name: CredentialName,
    body: EndTrialCredentialRequest,
    entities: NemoEntitiesClient = Depends(get_entity_client),
) -> TrialCredential:
    """Stop attributing the credential's calls after ``ended_at``. Ending it again at the same time returns it."""
    entity = await _get(entities, workspace, name)
    if entity.ended_at is not None:
        if entity.ended_at != body.ended_at:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="The credential already ended")
        return _to_wire(entity)
    if body.ended_at < entity.started_at:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="ended_at is before the trial was issued the credential",
        )
    return _to_wire(await entities.update(entity.model_copy(update={"ended_at": body.ended_at})))


@router.get(
    "/trial-credentials/{name}",
    summary="Get Trial Credential",
    include_in_schema=False,
    responses={status.HTTP_404_NOT_FOUND: {"description": "No trial holds the credential"}},
)
@scope.read
@path_rule(callers=[CallerKind.SERVICE_PRINCIPAL], permissions=[TrialCredentialPerms.READ])
async def get_trial_credential(
    workspace: str,
    name: CredentialName,
    entities: NemoEntitiesClient = Depends(get_entity_client),
) -> TrialCredential:
    """The trial a credential was issued to."""
    return _to_wire(await _get(entities, workspace, name))
