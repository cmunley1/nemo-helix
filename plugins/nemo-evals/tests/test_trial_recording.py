# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Each call made with a trial token is recorded in that trial, without touching the call."""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Mapping
from datetime import UTC, datetime
from importlib.metadata import entry_points
from typing import Any

import pytest
from nemo_evals import trial_recording
from nemo_evals.trial_recording import (
    MIDDLEWARE_NAME,
    CallApi,
    CallRecorder,
    FinishedCall,
    TokenUsage,
    TrialRecordingConfig,
    TrialRecordingMiddleware,
    caller_credential,
    usage_from_payload,
)
from nemo_evals.trial_spans import ATTEMPT_ATTRIBUTE, GATEWAY_CALL_ATTRIBUTE, SESSION_ATTRIBUTE, TEST_CASE_ATTRIBUTE
from nemo_evals.trial_tokens import ModelAccess, TrialClaims, TrialPolicy, TrialTokenReader, unsigned_trial_token
from nemo_helix_plugin.client.oidc import decode_jwt_claims
from nemo_helix_plugin.inference_middleware import InferenceMiddlewareContext, InferenceRequest, InferenceResponse
from opentelemetry.proto.trace.v1.trace_pb2 import ResourceSpans, Span

NOW = datetime.now(UTC)
TRIAL = TrialClaims(
    session="run-1:hello__abc",
    attempt=1,
    intake_workspace="evals",
    evaluation="run-1",
    task="hello",
    policy=TrialPolicy(model_access=ModelAccess.OPEN),
)
TOKEN = unsigned_trial_token(TRIAL)


def _value(value: Any) -> Any:
    return getattr(value, value.WhichOneof("value"))


def _attributes(span: Span) -> dict[str, Any]:
    return {attribute.key: _value(attribute.value) for attribute in span.attributes}


def _spans(batches: list[tuple[str, list[ResourceSpans]]]) -> list[tuple[str, Span]]:
    return [
        (workspace, span)
        for workspace, batch in batches
        for resource in batch
        for scope in resource.scope_spans
        for span in scope.spans
    ]


class Signed:
    """Stands in for the platform's signature check: trusts the claims of exactly the tokens it was given."""

    def __init__(self, *tokens: str) -> None:
        self._tokens = set(tokens)
        self.asked = 0

    async def __call__(self, token: str) -> Mapping[str, Any] | None:
        self.asked += 1
        return decode_jwt_claims(token) if token in self._tokens else None


class Sink:
    def __init__(self) -> None:
        self.sent: list[tuple[str, list[ResourceSpans]]] = []

    async def __call__(self, workspace: str, spans: list[ResourceSpans]) -> int:
        self.sent.append((workspace, spans))
        return 0


def _reader(*tokens: str) -> TrialTokenReader:
    return TrialTokenReader(Signed(*tokens))


def _middleware(reader: TrialTokenReader, sink: Sink) -> TrialRecordingMiddleware:
    middleware = TrialRecordingMiddleware()
    middleware._recorder = CallRecorder(reader, sink, flush_interval_sec=60)
    return middleware


def _ctx(headers: dict[str, str], path: str = "v1/chat/completions") -> InferenceMiddlewareContext:
    body = {"model": "default/agent-model", "messages": [{"role": "user", "content": "Hello"}]}
    return InferenceMiddlewareContext(
        request_id="req-1",
        virtual_model_name="agent-model",
        workspace="models",
        original_request=InferenceRequest(body=body, headers=headers, path=path),
        proxied_request=InferenceRequest(body={**body, "model": "meta/llama-3.3-70b"}, headers=headers, path=path),
    )


async def _call(middleware: TrialRecordingMiddleware, ctx: InferenceMiddlewareContext, result: Any) -> Any:
    config = TrialRecordingConfig()
    await middleware.process_request(ctx, ctx.original_request, config)
    response = await middleware.process_response(ctx, InferenceResponse(result=result, headers={}), config)
    return response.result


async def _stream(*chunks: dict[str, Any]) -> AsyncIterator[dict[str, Any]]:
    for chunk in chunks:
        yield chunk


def test_the_middleware_is_registered_for_the_gateway() -> None:
    entries = [e for e in entry_points(group="nemo.inference_middleware") if e.name == MIDDLEWARE_NAME]

    assert entries
    assert all(entry.load() is TrialRecordingMiddleware for entry in entries)


@pytest.mark.parametrize(
    ("headers", "credential"),
    [
        ({"authorization": "Bearer trial-token"}, "trial-token"),
        ({"authorization": "bearer  trial-token "}, "trial-token"),
        ({"x-api-key": "trial-token"}, "trial-token"),
        ({"authorization": "Basic dXNlcg==", "x-api-key": "trial-token"}, "trial-token"),
        ({"authorization": "Bearer "}, None),
        ({}, None),
    ],
)
def test_the_credential_is_the_bearer_token_or_else_the_api_key(
    headers: dict[str, str], credential: str | None
) -> None:
    assert caller_credential(headers) == credential


@pytest.mark.parametrize("auth_enabled", [True, False], ids=["auth on", "auth off"])
async def test_the_gateway_trusts_signed_tokens_with_auth_on_and_unsigned_ones_with_it_off(
    monkeypatch: pytest.MonkeyPatch, auth_enabled: bool
) -> None:
    """With auth on, an unsigned token naming a trial is ignored, even when the gateway accepted it."""
    monkeypatch.setattr(trial_recording, "platform_auth_enabled", lambda: auth_enabled)

    async def nothing_is_signed(token: str) -> None:
        return None

    monkeypatch.setattr(trial_recording, "verified_access_key_claims", nothing_is_signed)
    middleware = TrialRecordingMiddleware()
    monkeypatch.setattr(middleware, "_get_platform_client", lambda hook: object())
    monkeypatch.setattr(trial_recording, "intake_sink", lambda client: Sink())
    await middleware.on_startup()
    try:
        assert middleware._recorder is not None
        assert (await middleware._recorder._trials.trial(TOKEN) == TRIAL) is not auth_enabled
    finally:
        await middleware.on_shutdown()


async def test_a_call_with_a_token_naming_no_trial_is_never_checked() -> None:
    """Most callers aren't trials, so only tokens that claim a trial reach the signature check."""
    verify, sink = Signed(TOKEN), Sink()
    middleware = _middleware(TrialTokenReader(verify), sink)

    await _call(middleware, _ctx({"authorization": "Bearer users-own-key"}), {"usage": {"prompt_tokens": 1}})
    assert middleware._recorder is not None
    await middleware._recorder.flush()

    assert verify.asked == 0
    assert sink.sent == []


async def test_a_call_made_with_a_trial_token_is_recorded_in_the_trial() -> None:
    sink = Sink()
    middleware = _middleware(_reader(TOKEN), sink)
    body = {"choices": [], "usage": {"prompt_tokens": 120, "completion_tokens": 5}}

    assert await _call(middleware, _ctx({"authorization": f"Bearer {TOKEN}"}), body) is body
    assert middleware._recorder is not None
    await middleware._recorder.flush()

    [(workspace, span)] = _spans(sink.sent)
    assert workspace == "evals"
    assert span.trace_id.hex() == TRIAL.trace_id
    assert span.parent_span_id.hex() == TRIAL.root_span_id
    assert span.name == "chat.completions"
    attributes = _attributes(span)
    assert attributes[SESSION_ATTRIBUTE] == TRIAL.session
    assert attributes[TEST_CASE_ATTRIBUTE] == "hello"
    assert attributes[ATTEMPT_ATTRIBUTE] == 1
    assert attributes[GATEWAY_CALL_ATTRIBUTE] is True
    assert attributes["gen_ai.usage.input_tokens"] == 120
    assert attributes["gen_ai.usage.output_tokens"] == 5
    assert attributes["gen_ai.request.model"] == "meta/llama-3.3-70b"
    assert attributes["nemo.virtual_model"] == "models/agent-model"


@pytest.mark.parametrize(
    "headers",
    [
        pytest.param(
            {"authorization": f"Bearer {unsigned_trial_token(TRIAL.model_copy(update={'attempt': 2}))}"},
            id="a trial token the platform didn't sign",
        ),
        pytest.param({}, id="no credential"),
    ],
)
async def test_a_call_without_a_trusted_trial_token_is_not_recorded(headers: dict[str, str]) -> None:
    sink = Sink()
    middleware = _middleware(_reader(TOKEN), sink)

    await _call(middleware, _ctx(headers), {"usage": {"prompt_tokens": 1}})
    assert middleware._recorder is not None
    await middleware._recorder.flush()

    assert sink.sent == []


async def test_a_streamed_call_is_recorded_with_its_usage_once_the_caller_has_it_all() -> None:
    sink = Sink()
    middleware = _middleware(_reader(TOKEN), sink)
    chunks = [
        {"type": "message_start", "message": {"usage": {"input_tokens": 10, "cache_read_input_tokens": 30}}},
        {"type": "message_delta", "usage": {"output_tokens": 15}},
    ]

    stream = await _call(middleware, _ctx({"x-api-key": TOKEN}, path="v1/messages"), _stream(*chunks))
    assert middleware._recorder is not None
    assert middleware._recorder._queue.empty()
    assert [chunk async for chunk in stream] == chunks
    await middleware._recorder.flush()

    [(_, span)] = _spans(sink.sent)
    attributes = _attributes(span)
    assert span.name == "messages"
    assert attributes["gen_ai.usage.input_tokens"] == 40
    assert attributes["gen_ai.usage.output_tokens"] == 15
    assert "status" not in attributes


async def test_a_caller_that_stops_reading_mid_stream_is_recorded_as_cancelled() -> None:
    sink = Sink()
    middleware = _middleware(_reader(TOKEN), sink)

    stream = await _call(middleware, _ctx({"authorization": f"Bearer {TOKEN}"}), _stream({"a": 1}, {"b": 2}))
    assert await anext(stream) == {"a": 1}
    await stream.aclose()
    assert middleware._recorder is not None
    await middleware._recorder.flush()

    [(_, span)] = _spans(sink.sent)
    assert _attributes(span)["status"] == "cancelled"


async def test_the_span_starts_when_the_gateway_received_the_call() -> None:
    sink = Sink()
    middleware = _middleware(_reader(TOKEN), sink)
    ctx = _ctx({"authorization": f"Bearer {TOKEN}"})
    config = TrialRecordingConfig()

    await middleware.process_request(ctx, ctx.original_request, config)
    received = ctx.state(MIDDLEWARE_NAME).get("started_at")
    await middleware.process_response(ctx, InferenceResponse(result={}, headers={}), config)
    assert middleware._recorder is not None
    await middleware._recorder.flush()

    [(_, span)] = _spans(sink.sent)
    assert span.start_time_unix_nano == int(received.timestamp() * 1_000_000_000)


def _finished(token: str = TOKEN, name: str = "chat.completions") -> FinishedCall:
    return FinishedCall(token=token, name=name, started_at=NOW, ended_at=NOW, cancelled=False)


async def test_a_token_that_cant_be_checked_drops_the_call_and_keeps_recording() -> None:
    failed = False

    async def flaky(token: str) -> Mapping[str, Any] | None:
        nonlocal failed
        if not failed:
            failed = True
            raise RuntimeError("the platform's signing keys are unreachable")
        return decode_jwt_claims(token)

    sink = Sink()
    recorder = CallRecorder(TrialTokenReader(flaky), sink, flush_interval_sec=60)
    recorder.record(_finished(name="1"))
    recorder.record(_finished(name="2"))
    await recorder.flush()

    assert [span.name for _, span in _spans(sink.sent)] == ["2"]
    assert recorder.dropped == 1


# --- usage per API ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("api", "payloads", "expected"),
    [
        pytest.param(
            CallApi.CHAT_COMPLETIONS,
            [{"usage": {"prompt_tokens": 100, "completion_tokens": 7, "prompt_tokens_details": {"cached_tokens": 60}}}],
            TokenUsage(uncached_input_tokens=40, cached_input_tokens=60, output_tokens=7),
            id="chat-body",
        ),
        pytest.param(
            CallApi.CHAT_COMPLETIONS,
            [
                {"choices": [{"delta": {"content": "Hi"}}], "usage": None},
                {"choices": [], "usage": {"prompt_tokens": 12, "completion_tokens": 3}},
            ],
            TokenUsage(uncached_input_tokens=12, output_tokens=3),
            id="chat-stream-include-usage",
        ),
        pytest.param(
            CallApi.RESPONSES,
            [{"usage": {"input_tokens": 50, "output_tokens": 9, "input_tokens_details": {"cached_tokens": 20}}}],
            TokenUsage(uncached_input_tokens=30, cached_input_tokens=20, output_tokens=9),
            id="responses-body",
        ),
        pytest.param(
            CallApi.RESPONSES,
            [
                {"type": "response.output_text.delta", "delta": "Hi"},
                {
                    "type": "response.completed",
                    "response": {"usage": {"input_tokens": 40, "output_tokens": 4, "input_tokens_details": {}}},
                },
            ],
            TokenUsage(uncached_input_tokens=40, output_tokens=4),
            id="responses-stream",
        ),
        pytest.param(
            CallApi.MESSAGES,
            [
                {
                    "usage": {
                        "input_tokens": 10,
                        "cache_read_input_tokens": 30,
                        "cache_creation_input_tokens": 5,
                        "output_tokens": 8,
                    }
                }
            ],
            TokenUsage(uncached_input_tokens=10, cached_input_tokens=30, cache_write_input_tokens=5, output_tokens=8),
            id="messages-body",
        ),
        pytest.param(
            CallApi.MESSAGES,
            [
                {
                    "type": "message_start",
                    "message": {"usage": {"input_tokens": 10, "cache_read_input_tokens": 30, "output_tokens": 1}},
                },
                {"type": "content_block_delta", "delta": {"text": "Hi"}},
                {
                    "type": "message_delta",
                    "usage": {"input_tokens": 10, "cache_read_input_tokens": None, "output_tokens": 15},
                },
            ],
            TokenUsage(uncached_input_tokens=10, cached_input_tokens=30, output_tokens=15),
            id="messages-stream",
        ),
    ],
)
def test_usage_is_read_from_every_api_and_stream_shape(
    api: CallApi, payloads: list[dict[str, Any]], expected: TokenUsage
) -> None:
    """Every API's counts land in the same classes; a later chunk's null count keeps the earlier value."""
    usage = TokenUsage()
    for payload in payloads:
        if (reported := usage_from_payload(api, payload)) is not None:
            usage.merge(reported)
    assert usage == expected


def test_input_total_counts_every_class_once() -> None:
    usage = TokenUsage(uncached_input_tokens=10, cached_input_tokens=30, cache_write_input_tokens=5, output_tokens=8)
    assert usage.input_tokens == 45
    assert usage.attributes() == {
        "gen_ai.usage.input_tokens": 45,
        "nemo.usage.uncached_input_tokens": 10,
        "gen_ai.usage.cached_tokens": 30,
        "llm.token_count.prompt_details.cache_write": 5,
        "gen_ai.usage.output_tokens": 8,
    }


@pytest.mark.parametrize(
    ("path", "api"),
    [
        ("v1/chat/completions", CallApi.CHAT_COMPLETIONS),
        ("/v1/responses", CallApi.RESPONSES),
        ("v1/messages", CallApi.MESSAGES),
        ("v1/embeddings", CallApi.OTHER),
    ],
)
def test_api_is_read_from_the_path(path: str, api: CallApi) -> None:
    assert CallApi.from_path(path) is api


# --- the recorder -----------------------------------------------------------------------------


async def test_queued_calls_are_sent_in_one_batch_per_intake_workspace() -> None:
    other = unsigned_trial_token(TRIAL.model_copy(update={"intake_workspace": "other"}))
    sink = Sink()
    recorder = CallRecorder(_reader(TOKEN, other), sink, flush_interval_sec=0.05)
    recorder.start()
    recorder.record(_finished(name="1"))
    recorder.record(_finished(token=other, name="2"))
    recorder.record(_finished(name="3"))
    await asyncio.sleep(0.2)
    await recorder.stop()

    sent = sorted(
        (workspace, [span.name for _, span in _spans([(workspace, batch)])]) for workspace, batch in sink.sent
    )
    assert sent == [("evals", ["1", "3"]), ("other", ["2"])]


async def test_stopping_sends_the_calls_the_sender_was_still_collecting() -> None:
    """The sender holds a batch for up to the flush interval; shutting the gateway down mustn't lose it."""
    sink = Sink()
    recorder = CallRecorder(_reader(TOKEN), sink, flush_interval_sec=60)
    recorder.start()
    recorder.record(_finished(name="1"))
    recorder.record(_finished(name="2"))
    while recorder._queue.qsize():
        await asyncio.sleep(0.01)

    await recorder.stop()

    assert sorted(span.name for _, span in _spans(sink.sent)) == ["1", "2"]


async def test_a_batch_is_split_to_fit_intakes_body_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    sink = Sink()
    recorder = CallRecorder(_reader(TOKEN), sink, flush_interval_sec=60)
    for name in "12345":
        recorder.record(_finished(name=name))
    one_span = await recorder._span(_finished(name="1"))
    assert one_span is not None
    monkeypatch.setattr(trial_recording, "MAX_BATCH_BYTES", one_span[1].ByteSize() * 2)
    await recorder.stop()

    assert [[span.name for _, span in _spans([batch])] for batch in sink.sent] == [["1", "2"], ["3", "4"], ["5"]]


async def test_an_unreachable_intake_never_blocks_recording_and_drops_its_batches() -> None:
    """Recording only enqueues, so an Intake that hangs or fails costs the call nothing."""
    calls = 0

    async def hanging_sink(workspace: str, spans: list[ResourceSpans]) -> int:
        nonlocal calls
        calls += 1
        await asyncio.Event().wait()
        return 0

    recorder = CallRecorder(_reader(TOKEN), hanging_sink, max_queued_calls=2, flush_interval_sec=0.01)
    recorder.start()
    recorder.record(_finished())
    await asyncio.sleep(0.05)  # the sender is now stuck on that call's batch

    started = time.monotonic()
    for _ in range(5):
        recorder.record(_finished())
    assert time.monotonic() - started < 0.01
    assert recorder.dropped == 3  # two fit in the queue

    await recorder.stop(timeout_sec=0.05)
    assert calls >= 1


async def test_a_failing_intake_drops_the_batch_and_keeps_sending() -> None:
    sent: list[int] = []

    async def flaky_sink(workspace: str, spans: list[ResourceSpans]) -> int:
        if not sent:
            sent.append(0)
            raise RuntimeError("Intake is down")
        sent.append(len(spans))
        return 0

    recorder = CallRecorder(_reader(TOKEN), flaky_sink, flush_interval_sec=0.01)
    recorder.start()
    recorder.record(_finished())
    await asyncio.sleep(0.05)
    recorder.record(_finished())
    await asyncio.sleep(0.05)
    await recorder.stop()

    assert sent == [0, 1]
    assert recorder.dropped == 1


async def test_spans_intake_rejects_are_counted_as_dropped() -> None:
    async def picky_sink(workspace: str, spans: list[ResourceSpans]) -> int:
        return 1

    recorder = CallRecorder(_reader(TOKEN), picky_sink, flush_interval_sec=60)
    recorder.record(_finished())
    recorder.record(_finished())
    await recorder.flush()

    assert recorder.dropped == 1


async def test_stop_is_bounded_even_when_the_sink_ignores_cancellation() -> None:
    release = asyncio.Event()

    async def stubborn_sink(workspace: str, spans: list[ResourceSpans]) -> int:
        while not release.is_set():
            try:
                await asyncio.sleep(1)
            except asyncio.CancelledError:
                continue
        return 0

    recorder = CallRecorder(_reader(TOKEN), stubborn_sink, flush_interval_sec=0.01)
    recorder.start()
    recorder.record(_finished())
    await asyncio.sleep(0.05)

    started = time.monotonic()
    await recorder.stop(timeout_sec=0.1)
    assert time.monotonic() - started < 0.5
    release.set()
