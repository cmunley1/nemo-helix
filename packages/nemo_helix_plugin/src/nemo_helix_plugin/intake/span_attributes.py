# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Span attributes that producers and Intake agree on, so a model call's usage is counted once per session."""

GATEWAY_CALL_ATTRIBUTE = "nemo.inference_gateway.call"
"""``True`` on the span the Inference Gateway records for one upstream model call it served, after routing.

One such span per billed call, as the leaf: it carries the usage the provider reported. When the caller sent a W3C
``traceparent`` the span sits under the caller's span, in the caller's trace.
"""

USAGE_SUPERSEDED_ATTRIBUTE = "nemo.usage.superseded_by_gateway"
"""``True`` on the root span of a trace whose model calls the gateway recorded in the same session.

The trace's own usage repeats those calls, so only gateway call spans count toward its usage.
"""
