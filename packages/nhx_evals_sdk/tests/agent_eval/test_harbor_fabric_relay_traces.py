# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A Fabric agent with Relay brings Relay's OTLP trace back with its Harbor trial."""

import hashlib
import importlib.metadata
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, cast

import pytest

pytest.importorskip("harbor", reason="NemoFabricAgent subclasses Harbor's BaseAgent")

from harbor.environments.base import BaseEnvironment
from nemo_fabric.integrations.harbor.fabric_agent import FabricAgent
from nhx_evals_sdk.agent_eval.runtimes.fabric import otlp_receiver
from nhx_evals_sdk.agent_eval.runtimes.harbor import fabric_agent
from nhx_evals_sdk.agent_eval.runtimes.harbor.fabric_agent import (
    RELAY_CLI_DIR,
    RELAY_CLI_RELEASES_URL,
    RELAY_OTLP_PORT,
    RELAY_TRACES_DIR,
    NemoFabricAgent,
)
from nhx_evals_sdk.agent_eval.runtimes.harbor.fabric_installed_agent import FabricInstalledAgent
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest
from opentelemetry.proto.trace.v1.trace_pb2 import ResourceSpans, ScopeSpans, Span

RECEIVER_ENDPOINT = f"http://127.0.0.1:{RELAY_OTLP_PORT}/v1/traces"


def _config(*, relay: bool = True, opentelemetry: dict[str, Any] | None = None) -> dict[str, Any]:
    config: dict[str, Any] = {
        "metadata": {"name": "codex-agent"},
        "harness": {"adapter_id": "nvidia.fabric.codex"},
    }
    if relay:
        config["relay"] = {"observability": {"opentelemetry": opentelemetry} if opentelemetry else {}}
    return config


def _agent(tmp_path: Path, **config: Any) -> NemoFabricAgent:
    return NemoFabricAgent(logs_dir=tmp_path, fabric_config=_config(**config), fabric_workspace="/app")


class _Result:
    return_code = 0
    stdout = ""
    stderr = ""


class _Environment:
    """Records what the agent asks of the container, in order."""

    default_user = "agent"

    def __init__(self) -> None:
        self.events: list[str] = []
        self.uploaded: dict[str, str] = {}

    async def exec(self, command: str, **kwargs: Any) -> _Result:
        self.events.append(command)
        return _Result()

    async def upload_file(self, source_path: Path, target_path: str) -> None:
        self.uploaded[target_path] = Path(source_path).read_text(encoding="utf-8")

    async def upload_dir(self, source_path: Path, target_path: str) -> None:
        pass


def _endpoints(agent: NemoFabricAgent) -> list[str]:
    relay = agent._build_config().relay
    assert relay is not None and relay.observability is not None and relay.observability.opentelemetry is not None
    assert relay.observability.opentelemetry.enabled
    return [endpoint.endpoint for endpoint in relay.observability.opentelemetry.endpoints]


def test_relay_exports_its_trace_to_the_receiver_beside_the_harness(tmp_path: Path) -> None:
    assert _endpoints(_agent(tmp_path)) == [RECEIVER_ENDPOINT]


def test_the_configs_own_trace_destinations_are_kept_only_when_it_enabled_them(tmp_path: Path) -> None:
    own = {"type": "full", "endpoint": "http://collector:4318/v1/traces"}

    assert _endpoints(_agent(tmp_path, opentelemetry={"enabled": True, "endpoints": [own]})) == [
        own["endpoint"],
        RECEIVER_ENDPOINT,
    ]
    assert _endpoints(_agent(tmp_path, opentelemetry={"enabled": False, "endpoints": [own]})) == [RECEIVER_ENDPOINT]


def test_a_config_without_relay_gets_no_trace_capture(tmp_path: Path) -> None:
    agent = _agent(tmp_path, relay=False)

    assert agent._build_config().relay is None
    assert RELAY_CLI_DIR not in (agent._runner_path_prefix or "")


async def test_setup_installs_the_relay_cli_and_the_receiver(tmp_path: Path) -> None:
    agent, environment = _agent(tmp_path), _Environment()

    await agent.setup(environment)

    assert environment.uploaded["/tmp/nemo-fabric-otlp-receiver.py"] == Path(otlp_receiver.__file__).read_text()
    [install] = [command for command in environment.events if RELAY_CLI_RELEASES_URL in command]
    assert install.startswith(agent._runner_python) and RELAY_CLI_DIR in install
    assert (agent._runner_path_prefix or "").split(":")[0] == RELAY_CLI_DIR


async def test_setup_without_relay_installs_neither(tmp_path: Path) -> None:
    environment = _Environment()

    await _agent(tmp_path, relay=False).setup(environment)

    assert environment.uploaded == {}
    assert not [command for command in environment.events if RELAY_CLI_RELEASES_URL in command]


async def test_the_installed_agent_installs_them_too(tmp_path: Path) -> None:
    agent = FabricInstalledAgent(
        logs_dir=tmp_path,
        fabric_config=_config(),
        fabric_package="nemo-fabric[relay]==0.5.0",
        fabric_workspace="/app",
        fabric_default_max_turns=None,
    )
    environment = _Environment()

    await agent.install(cast(BaseEnvironment, environment))

    assert "/tmp/nemo-fabric-otlp-receiver.py" in environment.uploaded


@pytest.mark.parametrize("fails", [False, True], ids=["succeeds", "fails"])
async def test_the_receiver_listens_for_the_whole_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fails: bool
) -> None:
    async def fabric_run(self: Any, instruction: str, environment: _Environment, context: Any) -> None:
        environment.events.append("fabric run")
        if fails:
            raise RuntimeError("the harness failed")

    monkeypatch.setattr(FabricAgent, "run", fabric_run)
    environment = _Environment()

    if fails:
        with pytest.raises(RuntimeError):
            await _agent(tmp_path).run("say hello", environment, None)
    else:
        await _agent(tmp_path).run("say hello", environment, None)

    start, run, stop = environment.events
    assert f"/tmp/nemo-fabric-otlp-receiver.py {RELAY_TRACES_DIR} {RELAY_OTLP_PORT}" in start
    assert otlp_receiver.READY_FILENAME in start
    assert run == "fabric run"
    assert stop.startswith("kill ")


def test_after_the_run_the_exports_become_the_trials_otlp_trace(tmp_path: Path) -> None:
    traces = tmp_path / "traces"
    traces.mkdir()
    export = ExportTraceServiceRequest(
        resource_spans=[ResourceSpans(scope_spans=[ScopeSpans(spans=[Span(name="openai.responses")])])]
    )
    (traces / f"000001{otlp_receiver.EXPORT_SUFFIX}").write_bytes(export.SerializeToString())

    _agent(tmp_path).populate_context_post_run(_Context())

    assert "openai.responses" in (traces / "relay.jsonl").read_text()
    assert not list(traces.glob(f"*{otlp_receiver.EXPORT_SUFFIX}"))


class _Context:
    """Harbor's ``AgentContext``, as far as the post-run hook reads it."""

    def __init__(self) -> None:
        self.metadata: dict[str, Any] = {}

    def __getattr__(self, name: str) -> Any:
        return None


def _release(tmp_path: Path, *, checksum: str | None = None) -> tuple[str, Path]:
    """A local copy of a nemo-relay CLI release, laid out like GitHub's download URLs."""
    version = importlib.metadata.version("nemo-relay")
    arch = {"x86_64": "x86_64", "amd64": "x86_64", "aarch64": "aarch64", "arm64": "aarch64"}[platform.machine().lower()]
    asset = f"nemo-relay-cli-{arch}-unknown-linux-musl-{version}"
    release = tmp_path / "releases" / version
    release.mkdir(parents=True)
    (release / asset).write_bytes(b"#!/bin/sh\necho relay\n")
    digest = checksum or hashlib.sha256(b"#!/bin/sh\necho relay\n").hexdigest()
    (release / f"{asset}.sha256").write_text(f"{digest}  {asset}\n")
    return (tmp_path / "releases").as_uri(), tmp_path / "bin"


def _install(base_url: str, target: Path, path: str = "/nonexistent") -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", fabric_agent._RELAY_CLI_INSTALL, base_url, str(target)],
        capture_output=True,
        text=True,
        env={**os.environ, "PATH": path},
        check=False,
    )


def test_the_cli_matching_the_installed_bindings_is_installed(tmp_path: Path) -> None:
    base_url, target = _release(tmp_path)

    result = _install(base_url, target)

    assert result.returncode == 0, result.stderr
    assert (target / "nemo-relay").read_bytes() == b"#!/bin/sh\necho relay\n"
    assert os.access(target / "nemo-relay", os.X_OK)


def test_a_cli_that_doesnt_match_its_checksum_is_refused(tmp_path: Path) -> None:
    base_url, target = _release(tmp_path, checksum="0" * 64)

    result = _install(base_url, target)

    assert result.returncode != 0 and "sha256" in result.stderr
    assert not (target / "nemo-relay").exists()


def test_a_cli_already_on_path_is_kept(tmp_path: Path) -> None:
    existing = tmp_path / "existing"
    existing.mkdir()
    (existing / "nemo-relay").write_text("#!/bin/sh\n")
    (existing / "nemo-relay").chmod(0o755)

    result = _install((tmp_path / "missing").as_uri(), tmp_path / "bin", path=str(existing))

    assert result.returncode == 0, result.stderr
    assert not (tmp_path / "bin").exists()
