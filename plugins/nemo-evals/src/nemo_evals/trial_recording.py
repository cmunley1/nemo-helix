# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Inference Gateway middleware that records each model call an evaluation trial makes to Intake, in that trial.

An evaluation job issues each trial a trial token whose claims name the trial (see :mod:`nemo_evals.trial_tokens`).
For every call through a VirtualModel that lists this middleware in ``response_middleware``, the middleware notes
the caller's token, the token usage the provider reported, and when the caller had the whole response. In the
background it checks who issued the token and sends the call to Intake as a span in the trial's trace. With
platform auth on, it trusts only tokens the platform signed, whatever the gateway accepted; with auth off, only
unsigned tokens from ``nemo-evals``. A call made with any other credential is not recorded. Nothing here fails,
delays or changes the call: a full queue drops the call, and a failed check or send is logged and dropped.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import defaultdict
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from nemo_evals.trial_spans import CallerSpan, call_span, export_request
from nemo_evals.trial_tokens import TrialTokenReader, claims_a_trial
from nemo_helix_plugin.auth import platform_auth_enabled, verified_access_key_claims
from nemo_helix_plugin.client.client import AsyncNemoClient
from nemo_helix_plugin.inference_middleware import (
    InferenceMiddlewareContext,
    InferenceRequest,
    InferenceResponse,
    NemoInferenceMiddleware,
)
from nemo_helix_plugin.intake.client import AsyncIntakeClient
from opentelemetry.proto.trace.v1.trace_pb2 import ResourceSpans
from pydantic import BaseModel, ConfigDict, JsonValue

logger = logging.getLogger(__name__)

MIDDLEWARE_NAME = "nemo-evals-trial-recording"
CONFIG_TYPE = "trial_recording"
STARTED_AT_STATE_KEY = "started_at"

AGENT_HEADERS = ("x-claude-code-session-id", "session_id", "session-id", "x-codex-turn-metadata", "x-codex-window-id")
AGENT_HEADER_ATTRIBUTE_PREFIX = "nemo.agent.header."
MAX_HEADER_VALUE_LENGTH = 1024

DEFAULT_MAX_QUEUED_CALLS = 10_000
DEFAULT_MAX_BATCH_SIZE = 500
# Intake refuses OTLP bodies over 5 MiB.
MAX_BATCH_BYTES = 4 * 1024 * 1024
DEFAULT_FLUSH_INTERVAL_SEC = 1.0
DEFAULT_SHUTDOWN_FLUSH_TIMEOUT_SEC = 5.0
FAILURE_LOG_INTERVAL_SEC = 60.0


def caller_credential(headers: Mapping[str, str]) -> str | None:
    """The credential the caller sent: a bearer token, else an ``x-api-key``."""
    authorization = headers.get("authorization", "")
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() == "bearer" and token.strip():
        return token.strip()
    return headers.get("x-api-key", "").strip() or None


class CallApi(StrEnum):
    """The inference API a call used, from its path."""

    CHAT_COMPLETIONS = "chat.completions"
    RESPONSES = "responses"
    MESSAGES = "messages"
    OTHER = "other"

    @classmethod
    def from_path(cls, path: str) -> CallApi:
        path = path.strip("/").removeprefix("v1/")
        if path == "chat/completions":
            return cls.CHAT_COMPLETIONS
        if path == "responses":
            return cls.RESPONSES
        if path == "messages":
            return cls.MESSAGES
        return cls.OTHER


@dataclass
class TokenUsage:
    """Token counts for one call, by class: every input token is exactly one of uncached, cache read, or cache write."""

    uncached_input_tokens: int | None = None
    cached_input_tokens: int | None = None
    cache_write_input_tokens: int | None = None
    output_tokens: int | None = None

    def merge(self, other: TokenUsage) -> None:
        """Take every count *other* reports."""
        for name in ("uncached_input_tokens", "cached_input_tokens", "cache_write_input_tokens", "output_tokens"):
            value = getattr(other, name)
            if value is not None:
                setattr(self, name, value)

    @property
    def input_tokens(self) -> int | None:
        if self.uncached_input_tokens is None:
            return None
        return self.uncached_input_tokens + (self.cached_input_tokens or 0) + (self.cache_write_input_tokens or 0)

    def attributes(self) -> dict[str, JsonValue]:
        counts = {
            "gen_ai.usage.input_tokens": self.input_tokens,
            "nemo.usage.uncached_input_tokens": self.uncached_input_tokens,
            "gen_ai.usage.cached_tokens": self.cached_input_tokens,
            "llm.token_count.prompt_details.cache_write": self.cache_write_input_tokens,
            "gen_ai.usage.output_tokens": self.output_tokens,
        }
        return {key: value for key, value in counts.items() if value is not None}


def _count(mapping: Any, *path: str) -> int | None:
    for key in path:
        if not isinstance(mapping, Mapping):
            return None
        mapping = mapping.get(key)
    if isinstance(mapping, int) and not isinstance(mapping, bool):
        return mapping
    return None


def _openai_usage(total_input: int | None, cached: int | None, output: int | None) -> TokenUsage:
    uncached = None if total_input is None else total_input - (cached or 0)
    return TokenUsage(uncached_input_tokens=uncached, cached_input_tokens=cached, output_tokens=output)


def usage_from_payload(api: CallApi, payload: Mapping[str, Any]) -> TokenUsage | None:
    """The usage a response body or stream chunk reports, or ``None`` if it reports none."""
    if api is CallApi.MESSAGES:
        usage = payload.get("usage")
        if payload.get("type") == "message_start" and isinstance(payload.get("message"), Mapping):
            usage = payload["message"].get("usage")
        if not isinstance(usage, Mapping):
            return None
        return TokenUsage(
            uncached_input_tokens=_count(usage, "input_tokens"),
            cached_input_tokens=_count(usage, "cache_read_input_tokens"),
            cache_write_input_tokens=_count(usage, "cache_creation_input_tokens"),
            output_tokens=_count(usage, "output_tokens"),
        )
    usage = payload.get("usage")
    if api is CallApi.RESPONSES:
        if isinstance(payload.get("response"), Mapping):
            usage = payload["response"].get("usage")
        if not isinstance(usage, Mapping):
            return None
        return _openai_usage(
            _count(usage, "input_tokens"),
            _count(usage, "input_tokens_details", "cached_tokens"),
            _count(usage, "output_tokens"),
        )
    if not isinstance(usage, Mapping):
        return None
    return _openai_usage(
        _count(usage, "prompt_tokens"),
        _count(usage, "prompt_tokens_details", "cached_tokens"),
        _count(usage, "completion_tokens"),
    )


def model_from_payload(payload: Mapping[str, Any]) -> str | None:
    """The model a response body or stream chunk says served the call, or ``None`` if it names none."""
    for envelope in ("response", "message"):
        if isinstance(payload.get(envelope), Mapping):
            payload = payload[envelope]
            break
    model = payload.get("model")
    return model if isinstance(model, str) and model else None


@dataclass(frozen=True)
class FinishedCall:
    """A call the gateway served, waiting to be attributed to the trial its token names."""

    token: str
    name: str
    started_at: datetime
    ended_at: datetime
    cancelled: bool
    attributes: Mapping[str, object] = field(default_factory=dict)
    caller: CallerSpan | None = None


SpanSink = Callable[[str, list[ResourceSpans]], Awaitable[int]]
"""Sends a workspace's spans to Intake and returns how many Intake rejected."""


class CallRecorder:
    """Queues finished calls; in the background, attributes them to trials and sends them to Intake in batches."""

    def __init__(
        self,
        trials: TrialTokenReader,
        sink: SpanSink,
        *,
        max_queued_calls: int = DEFAULT_MAX_QUEUED_CALLS,
        max_batch_size: int = DEFAULT_MAX_BATCH_SIZE,
        flush_interval_sec: float = DEFAULT_FLUSH_INTERVAL_SEC,
    ) -> None:
        self._trials = trials
        self._sink = sink
        self._queue: asyncio.Queue[FinishedCall] = asyncio.Queue(maxsize=max_queued_calls)
        self._collecting: list[FinishedCall] = []
        self._sending: list[FinishedCall] = []
        self._flushing: list[FinishedCall] = []
        self._max_batch_size = max_batch_size
        self._flush_interval_sec = flush_interval_sec
        self._task: asyncio.Task[None] | None = None
        self._stopping = False
        self._dropped = 0
        self._last_failure_log = float("-inf")

    @property
    def dropped(self) -> int:
        """Calls dropped so far because the queue was full, a token couldn't be checked, or Intake rejected their batch."""
        return self._dropped

    def record(self, call: FinishedCall) -> None:
        """Queue *call*, dropping it if the queue is full."""
        try:
            self._queue.put_nowait(call)
        except asyncio.QueueFull:
            self._dropped += 1
            self._log_failure("Trial call recording queue is full; dropping calls")

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._run(), name="nemo-evals-trial-recorder")

    async def stop(self, timeout_sec: float = DEFAULT_SHUTDOWN_FLUSH_TIMEOUT_SEC) -> None:
        """Stop the sender, then send what is still queued; the whole stop takes at most *timeout_sec*.

        A batch the sender is already delivering gets the time to finish; one it can't finish counts as dropped.
        """
        deadline = time.monotonic() + timeout_sec
        self._stopping = True
        if self._task is not None:
            task, self._task = self._task, None
            if not self._sending:
                task.cancel()
            await asyncio.wait({task}, timeout=timeout_sec)
            if not task.done():
                self._dropped += len(self._sending)
                logger.warning("Dropped %d trial calls being sent at shutdown", len(self._sending))
                task.cancel()
        try:
            await asyncio.wait_for(self.flush(), timeout=max(0.0, deadline - time.monotonic()))
        except TimeoutError:
            lost = len(self._flushing) + self._queue.qsize()
            self._dropped += lost
            self._flushing = []
            logger.warning("Dropped %d queued trial calls at shutdown", lost)

    async def flush(self) -> None:
        """Send everything queued now, starting with any batch the sender was still collecting."""
        if self._collecting:
            self._flushing, self._collecting = self._collecting, []
            await self._send(self._flushing)
        while not self._queue.empty():
            self._flushing = self._take_batch()
            await self._send(self._flushing)
        self._flushing = []

    async def _run(self) -> None:
        while not self._stopping:
            try:
                await self._send_next_batch()
            except Exception:
                logger.exception("Trial call recording failed; continuing")

    async def _send_next_batch(self) -> None:
        first = await self._queue.get()
        deadline = time.monotonic() + self._flush_interval_sec
        self._collecting = [first]
        while len(self._collecting) < self._max_batch_size and (remaining := deadline - time.monotonic()) > 0:
            try:
                self._collecting.append(await asyncio.wait_for(self._queue.get(), timeout=remaining))
            except TimeoutError:
                break
        self._sending, self._collecting = self._collecting, []
        try:
            await self._send(self._sending)
        finally:
            self._sending = []

    def _take_batch(self) -> list[FinishedCall]:
        batch: list[FinishedCall] = []
        while len(batch) < self._max_batch_size and not self._queue.empty():
            batch.append(self._queue.get_nowait())
        return batch

    async def _span(self, call: FinishedCall) -> tuple[str, ResourceSpans] | None:
        try:
            trial = await self._trials.trial(call.token)
        except Exception:
            self._dropped += 1
            self._log_failure("Could not check a caller's trial token", exc_info=True)
            return None
        if trial is None:
            return None
        attributes = {**call.attributes, "status": "cancelled"} if call.cancelled else call.attributes
        span = call_span(
            trial,
            name=call.name,
            started_at=call.started_at,
            ended_at=call.ended_at,
            error=False,
            attributes=attributes,
            caller=call.caller,
        )
        return trial.intake_workspace, span

    async def _send(self, batch: list[FinishedCall]) -> None:
        by_workspace: dict[str, list[ResourceSpans]] = defaultdict(list)
        for call in batch:
            if (attributed := await self._span(call)) is not None:
                by_workspace[attributed[0]].append(attributed[1])
        for workspace, spans in by_workspace.items():
            for chunk in _chunks_under(spans, MAX_BATCH_BYTES):
                try:
                    rejected = await self._sink(workspace, chunk)
                except Exception:
                    self._dropped += len(chunk)
                    self._log_failure(
                        f"Could not send {len(chunk)} trial call spans to Intake for workspace {workspace!r}",
                        exc_info=True,
                    )
                    continue
                if rejected:
                    self._dropped += rejected
                    self._log_failure(f"Intake rejected {rejected} trial call spans for workspace {workspace!r}")

    def _log_failure(self, message: str, *, exc_info: bool = False) -> None:
        now = time.monotonic()
        if now - self._last_failure_log >= FAILURE_LOG_INTERVAL_SEC:
            self._last_failure_log = now
            logger.warning("%s (%d dropped so far)", message, self._dropped, exc_info=exc_info)


def _chunks_under(spans: list[ResourceSpans], max_bytes: int) -> list[list[ResourceSpans]]:
    chunks: list[list[ResourceSpans]] = [[]]
    size = 0
    for span in spans:
        span_size = span.ByteSize()
        if chunks[-1] and size + span_size > max_bytes:
            chunks.append([])
            size = 0
        chunks[-1].append(span)
        size += span_size
    return chunks


def intake_sink(client: AsyncNemoClient) -> SpanSink:
    """Send each workspace's spans to Intake's OTLP route with *client*."""
    intake = AsyncIntakeClient.from_client(client)

    async def sink(workspace: str, spans: list[ResourceSpans]) -> int:
        response = (await intake.create_otlp_traces(workspace=workspace, content=export_request(spans))).data()
        if response.errors:
            logger.debug("Intake rejected trial call spans for workspace %r: %s", workspace, response.errors[:5])
        return len(response.errors)

    return sink


class TrialRecordingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class _ObservedCall:
    """One call through the middleware: what it was, and the usage its response reported."""

    def __init__(self, ctx: InferenceMiddlewareContext, token: str) -> None:
        request = ctx.original_request
        self.ctx = ctx
        self.token = token
        self.api = CallApi.from_path(request.path)
        self.usage = TokenUsage()
        self.response_model: str | None = None
        self.agent_headers = {
            name: value[:MAX_HEADER_VALUE_LENGTH] for name in AGENT_HEADERS if (value := request.headers.get(name))
        }
        self.caller = CallerSpan.from_headers(request.headers.get("traceparent"), request.headers.get("tracestate"))

    def observe(self, payload: object) -> None:
        if not isinstance(payload, Mapping):
            return
        if (usage := usage_from_payload(self.api, payload)) is not None:
            self.usage.merge(usage)
        self.response_model = model_from_payload(payload) or self.response_model

    def finished(self, started_at: datetime, *, cancelled: bool) -> FinishedCall:
        proxied = self.ctx.proxied_request
        model = (proxied.body if proxied is not None else self.ctx.original_request.body).get("model")
        attributes: dict[str, object] = {
            "gen_ai.operation.name": "chat",
            "nemo.virtual_model": f"{self.ctx.workspace}/{self.ctx.virtual_model_name}",
            "nemo.inference.api": self.api.value,
            **{f"{AGENT_HEADER_ATTRIBUTE_PREFIX}{name}": value for name, value in self.agent_headers.items()},
            **self.usage.attributes(),
        }
        if isinstance(model, str):
            attributes["gen_ai.request.model"] = model
        if self.response_model is not None:
            attributes["gen_ai.response.model"] = self.response_model
        return FinishedCall(
            token=self.token,
            name=self.api.value,
            started_at=started_at,
            ended_at=datetime.now(UTC),
            cancelled=cancelled,
            attributes=attributes,
            caller=self.caller,
        )


class TrialRecordingMiddleware(NemoInferenceMiddleware):
    """Record each call made with a trial token to Intake, in the trial.

    It records from ``response_middleware``. Listed in ``request_middleware`` too, a call's span starts when the
    gateway received the call rather than when the response began.
    """

    def __init__(self) -> None:
        super().__init__()
        self._recorder: CallRecorder | None = None

    def supports_middleware_phase(self, phase: str) -> bool:
        return phase in {"request", "response"}

    async def validate_middleware_config(self, config_type: str, config: Any) -> TrialRecordingConfig:
        if config_type != CONFIG_TYPE:
            raise ValueError(f"{MIDDLEWARE_NAME} supports config_type {CONFIG_TYPE!r}, not {config_type!r}")
        return TrialRecordingConfig.model_validate(config or {})

    async def on_startup(self) -> None:
        client = self._get_platform_client("on_startup")
        reader = TrialTokenReader(verified_access_key_claims if platform_auth_enabled() else None)
        self._recorder = CallRecorder(reader, intake_sink(client))
        self._recorder.start()

    async def on_shutdown(self) -> None:
        if self._recorder is not None:
            await self._recorder.stop()

    async def process_request(
        self,
        ctx: InferenceMiddlewareContext,
        request: InferenceRequest,
        middleware_config: TrialRecordingConfig,
    ) -> InferenceRequest:
        ctx.state(MIDDLEWARE_NAME).set(STARTED_AT_STATE_KEY, datetime.now(UTC))
        return request

    async def process_response(
        self,
        ctx: InferenceMiddlewareContext,
        response: InferenceResponse,
        middleware_config: TrialRecordingConfig,
    ) -> InferenceResponse:
        recorder = self._recorder
        credential = caller_credential(ctx.original_request.headers)
        if recorder is None or credential is None or not claims_a_trial(credential):
            return response
        call = _ObservedCall(ctx, credential)
        started_at = ctx.state(MIDDLEWARE_NAME).get(STARTED_AT_STATE_KEY) or datetime.now(UTC)
        if isinstance(response.result, Mapping):
            call.observe(response.result)
            recorder.record(call.finished(started_at, cancelled=False))
            return response
        response.result = _follow(response.result, call, recorder, started_at)
        return response


async def _follow(
    chunks: AsyncIterator[dict[str, Any]], call: _ObservedCall, recorder: CallRecorder, started_at: datetime
) -> AsyncIterator[dict[str, Any]]:
    """Pass the stream through, recording the call when the caller has all of it or stops reading."""
    cancelled = True
    try:
        async for chunk in chunks:
            call.observe(chunk)
            yield chunk
        cancelled = False
    finally:
        recorder.record(call.finished(started_at, cancelled=cancelled))
