# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Enforces the model policy an evals credential carries on every Inference Gateway call.

A trial token or router key lists the VirtualModels it may call, and a closed policy requires each to be closed-book
(see :mod:`nemo_evals.trial_tokens`). The gateway calls :meth:`ModelPolicyEnforcer.check` for every call on every
route before any VirtualModel's middleware runs, so a caller can't reach a model by a route its policy doesn't name.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import NoReturn

from nemo_evals.closed_book import MIDDLEWARE_NAME as CLOSED_BOOK_MIDDLEWARE
from nemo_evals.trial_tokens import (
    ModelAccess,
    ModelPolicy,
    TrialTokenReader,
    caller_credential,
    claims_evals_credential,
)
from nemo_helix_plugin.inference_middleware import (
    InferenceAccessRequest,
    InferenceMiddlewareError,
    InferenceMiddlewareUnavailableError,
    MiddlewareCall,
    VirtualModel,
)

logger = logging.getLogger(__name__)

TRIAL_RECORDING_MIDDLEWARE = "nemo-evals-trial-recording"
LORA_ADAPTER_SEPARATOR = "&adapters/"


class RefusalReason(StrEnum):
    PROVIDER_ROUTE = "provider_route"
    UNLISTED_MODEL = "unlisted_model"
    UNPINNED_MODEL = "unpinned_model"
    NOT_CLOSED_BOOK = "not_closed_book"
    UNVERIFIED_CREDENTIAL = "unverified_credential"
    MISSING_CREDENTIAL = "missing_credential"


@dataclass(frozen=True)
class Refusal:
    reason: RefusalReason
    detail: str
    status_code: int = 403


def ends_closed_book(request_middleware: Sequence[MiddlewareCall]) -> bool:
    """Whether a VirtualModel's request middleware ends with the closed-book middleware."""
    return bool(request_middleware) and request_middleware[-1].name == CLOSED_BOOK_MIDDLEWARE


def policy_refusal(
    policy: ModelPolicy, request: InferenceAccessRequest, virtual_model: VirtualModel | None
) -> Refusal | None:
    """Why ``policy`` refuses ``request``, served by ``virtual_model``; ``None`` when it allows it."""
    allowed = ", ".join(policy.models) if policy.models else "any VirtualModel"
    if request.provider_name is not None:
        return Refusal(
            RefusalReason.PROVIDER_ROUTE,
            f"This credential may call only {allowed}, not the provider route to "
            f"{request.workspace}/{request.provider_name}, which skips every VirtualModel's middleware.",
        )
    model = request.virtual_model_id
    if policy.models is not None and model not in policy.models:
        return Refusal(RefusalReason.UNLISTED_MODEL, f"This credential may call only {allowed}, not {model}.")
    if policy.models is not None and LORA_ADAPTER_SEPARATOR in (request.requested_model or ""):
        return Refusal(
            RefusalReason.UNLISTED_MODEL,
            f"This credential may call only {allowed}, not the LoRA adapter {request.requested_model}.",
        )
    if policy.models is not None and (virtual_model is None or not virtual_model.default_model_entity):
        return Refusal(
            RefusalReason.UNPINNED_MODEL,
            f"{model} sets no default_model_entity, so its caller could name any model through it.",
        )
    if policy.model_access is ModelAccess.CLOSED and (
        virtual_model is None or not ends_closed_book(virtual_model.request_middleware)
    ):
        return Refusal(
            RefusalReason.NOT_CLOSED_BOOK,
            f"This credential may call only closed-book models, and {model} doesn't end its request_middleware "
            f"with {CLOSED_BOOK_MIDDLEWARE}.",
        )
    return None


def _records_trials(virtual_model: VirtualModel) -> bool:
    return any(
        call.name == TRIAL_RECORDING_MIDDLEWARE
        for call in [*virtual_model.request_middleware, *virtual_model.response_middleware]
    )


RefusalSink = Callable[[str, InferenceAccessRequest, Refusal], None]
"""Told of each call refused for a credential's policy: the credential, the call, and why."""


class ModelPolicyEnforcer:
    """Refuses each gateway call its caller's evals credential doesn't allow.

    A call without an evals credential is left alone, except that with platform auth off a call to a VirtualModel
    that records trials must carry one: an agent can't escape its policy there by dropping its token. Closed-book
    VirtualModels without recording also serve model evaluations and judges, which carry no evals credential.
    With auth off nothing is signed, so a caller that writes its own token, or leaves it off on any other route, isn't
    stopped.
    """

    def __init__(
        self,
        reader: TrialTokenReader,
        virtual_model: Callable[[str], VirtualModel | None],
        *,
        auth_enabled: bool,
        on_refused: RefusalSink | None = None,
    ) -> None:
        self._reader = reader
        self._virtual_model = virtual_model
        self._auth_enabled = auth_enabled
        self._on_refused = on_refused

    async def check(self, request: InferenceAccessRequest) -> None:
        """Return when ``request`` may proceed.

        Raises:
            InferenceMiddlewareError: The caller's credential doesn't allow the call.
            InferenceMiddlewareUnavailableError: The credential couldn't be verified.
        """
        credential = caller_credential(request.headers)
        model = request.virtual_model_id
        if credential is None or not claims_evals_credential(credential):
            if not self._auth_enabled and model is not None:
                virtual_model = self._virtual_model(model)
                if virtual_model is not None and _records_trials(virtual_model):
                    self._refuse(
                        None,
                        request,
                        Refusal(
                            RefusalReason.MISSING_CREDENTIAL,
                            f"{model} records evaluation trials, so calls to it must carry the credential the "
                            "evaluation issued.",
                            status_code=401,
                        ),
                    )
            return
        try:
            claims = await self._reader.claims(credential)
        except Exception as exc:
            logger.warning("Couldn't verify an evals credential", exc_info=True)
            raise InferenceMiddlewareUnavailableError("The caller's evals credential couldn't be verified.") from exc
        if claims is None:
            self._refuse(
                None,
                request,
                Refusal(
                    RefusalReason.UNVERIFIED_CREDENTIAL,
                    "The caller's credential carries an evals model policy the platform didn't issue.",
                ),
            )
        refusal = policy_refusal(claims, request, self._virtual_model(model) if model is not None else None)
        if refusal is not None:
            self._refuse(credential, request, refusal)

    def _refuse(self, credential: str | None, request: InferenceAccessRequest, refusal: Refusal) -> NoReturn:
        logger.warning(
            "Refused a gateway call for its caller's model policy",
            extra={
                "request_id": request.request_id,
                "workspace": request.workspace,
                "virtual_model": request.virtual_model_id,
                "provider": request.provider_name,
                "reason": refusal.reason.value,
            },
        )
        if credential is not None and self._on_refused is not None:
            self._on_refused(credential, request, refusal)
        raise InferenceMiddlewareError(refusal.detail, status_code=refusal.status_code)
