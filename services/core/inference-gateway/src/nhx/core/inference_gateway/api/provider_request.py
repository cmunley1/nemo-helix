# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared provider request policy helpers."""

import re

_DEFAULT_AUTH_HEADER_FORMAT = "Authorization: Bearer {{ auth_secret }}"

# ``auth_header_format`` is literal text around exactly one ``{{ auth_secret }}`` placeholder. It is
# never evaluated by a template engine: providers stored before this was enforced may carry hostile
# templates, so the format is re-validated here on every use. Keep in sync with the models service
# validator (``nhx.core.models.schemas._AUTH_HEADER_FORMAT_RE``). The ``<Name>: <Value>`` shape is
# additionally enforced in :func:`render_auth_header`.
_AUTH_HEADER_FORMAT_RE = re.compile(r"([^{}\r\n]*)\{\{\s*auth_secret\s*\}\}([^{}\r\n]*)")


_SECRET_SENTINEL = "\x00"


def render_auth_header(secret_value: str, auth_header_format: str | None) -> tuple[str, str]:
    """Build a provider authentication header from its ``auth_header_format``.

    The format must be ``<Name>: <Value>`` where exactly one ``{{ auth_secret }}`` placeholder
    appears in the value; the header name must be non-empty literal text.

    Raises:
        ValueError: If the format is not literal text with exactly one ``{{ auth_secret }}``
            placeholder, or does not have the ``<Name>: <Value>`` shape with the placeholder in
            the value. Neither the format nor the secret is echoed back.
    """
    template = auth_header_format or _DEFAULT_AUTH_HEADER_FORMAT
    match = _AUTH_HEADER_FORMAT_RE.fullmatch(template)
    if match is None:
        raise ValueError(
            "Provider auth_header_format is invalid: it must contain exactly one '{{ auth_secret }}' placeholder"
        )
    # Split the shape on the literal text with a sentinel in place of the secret so a secret that
    # happens to contain ": " can never move the name/value boundary.
    header_name, separator, header_value = (match.group(1) + _SECRET_SENTINEL + match.group(2)).partition(": ")
    if not separator or not header_name.strip() or _SECRET_SENTINEL in header_name:
        raise ValueError("Provider auth_header_format is invalid: it must have the form '<Name>: <Value>'")
    return header_name, header_value.replace(_SECRET_SENTINEL, secret_value)
