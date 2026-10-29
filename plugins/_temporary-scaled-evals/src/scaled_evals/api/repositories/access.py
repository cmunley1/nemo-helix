# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import LiteralString


@dataclass(frozen=True)
class ResourceAccess:
    owner_id: str
    include_all: bool

    @property
    def parameters(self) -> tuple[str, ...]:
        return () if self.include_all else (self.owner_id,)

    def read_clause(self, alias: LiteralString = "") -> LiteralString:
        if self.include_all:
            return "TRUE"
        prefix = f"{alias}." if alias else ""
        return f"({prefix}owner_id = %s OR {prefix}visibility IN ('public', 'team', 'org'))"

    def write_clause(self, alias: LiteralString = "") -> LiteralString:
        if self.include_all:
            return "TRUE"
        prefix = f"{alias}." if alias else ""
        return f"{prefix}owner_id = %s"

    @property
    def run_parameters(self) -> tuple[str, ...]:
        return self.parameters * 2

    def run_read_clause(self, alias: LiteralString = "benchmark_runs") -> LiteralString:
        if self.include_all:
            return "TRUE"
        return (
            f"({self.read_clause(alias)} AND NOT EXISTS (SELECT 1 FROM evaluations inaccessible "
            f"WHERE inaccessible.benchmark_run_id = {alias}.id AND inaccessible.deleted_at IS NULL "
            f"AND NOT ({self.read_clause('inaccessible')})))"
        )


INTERNAL_ACCESS = ResourceAccess(owner_id="", include_all=True)
