# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for local dataset path resolution in the unsloth SFT driver.

These exercise ``_resolve_local_data_files`` only, which uses stdlib
``pathlib`` and pulls in no heavy ML deps — so the module imports fine on
a CPU box without ``unsloth``/``torch`` installed.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from nhx.customization_common.dataset_prepare import (
    SPLIT_FILES,
    DatasetFormat,
    map_columns,
    split_rows,
    to_training_rows,
    write_split,
)
from nhx.unsloth.tasks.training.backends.unsloth_sft import _resolve_local_data_files


def test_single_file_passthrough(tmp_path: Path) -> None:
    f = tmp_path / "train.jsonl"
    f.write_text('{"text": "hi"}\n')
    assert _resolve_local_data_files(str(f)) == str(f)


def test_directory_expands_to_jsonl(tmp_path: Path) -> None:
    """Container submit hands the loader the download dir, not a file."""
    (tmp_path / "train.jsonl").write_text('{"text": "hi"}\n')
    assert _resolve_local_data_files(str(tmp_path)) == [str(tmp_path / "train.jsonl")]


def test_directory_prefers_jsonl_over_json(tmp_path: Path) -> None:
    (tmp_path / "a.jsonl").write_text("{}\n")
    (tmp_path / "b.json").write_text("{}\n")
    assert _resolve_local_data_files(str(tmp_path)) == [str(tmp_path / "a.jsonl")]


def test_directory_falls_back_to_json(tmp_path: Path) -> None:
    (tmp_path / "data.json").write_text("{}\n")
    assert _resolve_local_data_files(str(tmp_path)) == [str(tmp_path / "data.json")]


def test_directory_returns_sorted_files(tmp_path: Path) -> None:
    for name in ("c.jsonl", "a.jsonl", "b.jsonl"):
        (tmp_path / name).write_text("{}\n")
    assert _resolve_local_data_files(str(tmp_path)) == [
        str(tmp_path / "a.jsonl"),
        str(tmp_path / "b.jsonl"),
        str(tmp_path / "c.jsonl"),
    ]


def test_empty_directory_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="No .jsonl/.json files"):
        _resolve_local_data_files(str(tmp_path))


def test_split_train_picks_train_jsonl_only(tmp_path: Path) -> None:
    (tmp_path / "train.jsonl").write_text('{"text": "train"}\n')
    (tmp_path / "validation.jsonl").write_text('{"text": "val"}\n')
    assert _resolve_local_data_files(str(tmp_path), split="train") == str(tmp_path / "train.jsonl")


def test_split_validation_picks_validation_jsonl_only(tmp_path: Path) -> None:
    (tmp_path / "train.jsonl").write_text('{"text": "train"}\n')
    (tmp_path / "validation.jsonl").write_text('{"text": "val"}\n')
    assert _resolve_local_data_files(str(tmp_path), split="validation") == str(
        tmp_path / "validation.jsonl",
    )


def _prepared(tmp_path: Path, dataset_format: DatasetFormat, rows: list[dict], **named: str) -> Path:
    """Write *rows* the way ``nemo customization dataset prepare --format`` uploads them."""
    training_rows, _ = to_training_rows(rows, map_columns(dataset_format, rows[0].keys(), named))
    splits = split_rows(training_rows, val_fraction=0.2, test_fraction=0.0, seed=0)
    write_split(splits.train, tmp_path / SPLIT_FILES["train"])
    write_split(splits.validation, tmp_path / SPLIT_FILES["validation"])
    return tmp_path


def test_data_designer_instruction_output_loads_each_split_as_messages(tmp_path: Path) -> None:
    """The Studio "Instruction fine-tuning" template, prepared by the CLI, loads with apply_chat_template."""
    import json

    rows = [{"domain": "d", "instruction": f"q{i}", "response": f"a{i}"} for i in range(10)]
    dataset = _prepared(tmp_path, DatasetFormat.CHAT, rows)

    train = _resolve_local_data_files(str(dataset), split="train")
    validation = _resolve_local_data_files(str(dataset), split="validation")

    assert (train, validation) == (str(dataset / "train.jsonl"), str(dataset / "validation.jsonl"))
    first = json.loads((dataset / "train.jsonl").read_text().splitlines()[0])
    assert [message["role"] for message in first["messages"]] == ["user", "assistant"]
