# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for provider auth header construction (PSIRT: no template evaluation)."""

from pathlib import Path

import pytest
from nhx.core.inference_gateway.api.provider_request import render_auth_header


@pytest.mark.parametrize(
    ("fmt", "expected"),
    [
        (None, ("Authorization", "Bearer s3cret")),
        ("X-Api-Key: {{ auth_secret }}", ("X-Api-Key", "s3cret")),
        ("X-Api-Key: {{auth_secret}}", ("X-Api-Key", "s3cret")),
        ("X-Custom: pre-{{ auth_secret }}-post", ("X-Custom", "pre-s3cret-post")),
    ],
)
def test_builds_header_from_valid_format(fmt: str | None, expected: tuple[str, str]) -> None:
    assert render_auth_header("s3cret", fmt) == expected


def test_secret_containing_template_syntax_is_inserted_literally() -> None:
    assert render_auth_header("{{ 7*7 }}", "X-Api-Key: {{ auth_secret }}") == ("X-Api-Key", "{{ 7*7 }}")


@pytest.mark.parametrize(
    "fmt",
    [
        "X-Api-Key: {{ auth_secret }}{{ ''.__class__ }}",
        "X-Api-Key: {{ auth_secret | upper }}",
        "X-Api-Key: {{ auth_secret }}{% if 1 %}x{% endif %}",
        "X-Api-Key: {{ other }}",
        "X-Api-Key: static",
        "X-Api-Key: {{ auth_secret }} {{ auth_secret }}",
    ],
)
def test_rejects_stored_hostile_or_malformed_format(fmt: str) -> None:
    with pytest.raises(ValueError, match="auth_header_format is invalid"):
        render_auth_header("s3cret", fmt)


def test_hostile_stored_format_is_never_executed(tmp_path: Path) -> None:
    marker = tmp_path / "ssti_rce_proof"
    fmt = (
        "Authorization: Bearer {{ auth_secret }}{{ (''.__class__.__mro__[1].__subclasses__() | "
        "selectattr('__name__', 'equalto', '_wrap_close') | first).__init__.__globals__['popen']"
        f"('id > {marker}').read() }}}}"
    )
    with pytest.raises(ValueError):
        render_auth_header("s3cret", fmt)
    assert not marker.exists()


@pytest.mark.parametrize(
    "fmt",
    [
        "{{ auth_secret }}",  # no ': ' separator: secret would become the header name
        "Bearer {{ auth_secret }}",
        ": {{ auth_secret }}",  # empty header name
        " : {{ auth_secret }}",
        "{{ auth_secret }}: value",  # secret in the header name
    ],
)
def test_rejects_format_without_name_value_shape(fmt: str) -> None:
    with pytest.raises(ValueError, match="'<Name>: <Value>'") as exc_info:
        render_auth_header("s3cret", fmt)
    assert "s3cret" not in str(exc_info.value)


def test_secret_containing_separator_does_not_move_name_value_boundary() -> None:
    assert render_auth_header("a: b", "X-Api-Key: {{ auth_secret }}") == ("X-Api-Key", "a: b")
