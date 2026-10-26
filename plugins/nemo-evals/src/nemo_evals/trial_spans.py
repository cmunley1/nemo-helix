# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OTLP spans in one trial's trace: the trial's root, and a child span per model call the gateway served it."""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Mapping
from datetime import datetime

from nemo_helix_plugin.evals.types import TrialCredential
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest
from opentelemetry.proto.common.v1.common_pb2 import AnyValue, ArrayValue, InstrumentationScope, KeyValue
from opentelemetry.proto.resource.v1.resource_pb2 import Resource
from opentelemetry.proto.trace.v1.trace_pb2 import ResourceSpans, ScopeSpans, Span, Status

SERVICE_NAME = "nemo-evals"
SESSION_ATTRIBUTE = "gen_ai.conversation.id"
EVALUATION_ATTRIBUTE = "nemo.evaluation.name"
TEST_CASE_ATTRIBUTE = "nemo.test_case.name"
# Marks a call span recorded by the gateway, whose usage is what the provider reported for the call.
GATEWAY_CALL_ATTRIBUTE = "nemo.inference_gateway.call"

_SCOPE = InstrumentationScope(name="nemo_evals.trial_spans")


def _any_value(value: object) -> AnyValue:
    if isinstance(value, bool):
        return AnyValue(bool_value=value)
    if isinstance(value, int):
        return AnyValue(int_value=value)
    if isinstance(value, float):
        return AnyValue(double_value=value)
    if isinstance(value, str):
        return AnyValue(string_value=value)
    if isinstance(value, list | tuple) and all(isinstance(item, str) for item in value):
        return AnyValue(array_value=ArrayValue(values=[AnyValue(string_value=item) for item in value]))
    return AnyValue(string_value=json.dumps(value, default=str))


def _nanos(at: datetime) -> int:
    return int(at.timestamp() * 1_000_000_000)


def _trial_attributes(trial: TrialCredential) -> dict[str, object]:
    attributes: dict[str, object] = {SESSION_ATTRIBUTE: trial.session_id}
    if trial.evaluation_name:
        attributes[EVALUATION_ATTRIBUTE] = trial.evaluation_name
    if trial.test_case_name:
        attributes[TEST_CASE_ATTRIBUTE] = trial.test_case_name
    return attributes


def _resource_spans(span: Span, attributes: Mapping[str, object]) -> ResourceSpans:
    for key, value in attributes.items():
        if value is not None:
            span.attributes.add(key=key, value=_any_value(value))
    resource = Resource(attributes=[KeyValue(key="service.name", value=_any_value(SERVICE_NAME))])
    return ResourceSpans(resource=resource, scope_spans=[ScopeSpans(scope=_SCOPE, spans=[span])])


def export_request(batch: Iterable[ResourceSpans]) -> bytes:
    return ExportTraceServiceRequest(resource_spans=list(batch)).SerializeToString()


def call_span(
    trial: TrialCredential,
    *,
    name: str,
    started_at: datetime,
    ended_at: datetime,
    error: bool,
    attributes: Mapping[str, object],
) -> ResourceSpans:
    """One model call the gateway served the trial, as a child of the trial's root."""
    span = Span(
        trace_id=bytes.fromhex(trial.trace_id),
        span_id=os.urandom(8),
        parent_span_id=bytes.fromhex(trial.root_span_id),
        name=name,
        kind=Span.SPAN_KIND_SERVER,
        start_time_unix_nano=_nanos(started_at),
        end_time_unix_nano=_nanos(ended_at),
        status=Status(code=Status.STATUS_CODE_ERROR if error else Status.STATUS_CODE_OK),
    )
    return _resource_spans(span, {**attributes, GATEWAY_CALL_ATTRIBUTE: True, **_trial_attributes(trial)})


def trial_root_span(trial: TrialCredential) -> ResourceSpans:
    """The trial's root span, from when it was issued its credential until the credential ended.

    Raises:
        ValueError: The credential has not ended.
    """
    if trial.ended_at is None:
        raise ValueError("a trial's root span needs the trial's credential to have ended")
    span = Span(
        trace_id=bytes.fromhex(trial.trace_id),
        span_id=bytes.fromhex(trial.root_span_id),
        name="trial",
        kind=Span.SPAN_KIND_INTERNAL,
        start_time_unix_nano=_nanos(trial.started_at),
        end_time_unix_nano=_nanos(trial.ended_at),
        status=Status(code=Status.STATUS_CODE_OK),
    )
    return _resource_spans(span, {**_trial_attributes(trial), "gen_ai.operation.name": "invoke_agent"})
