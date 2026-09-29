#===- act/front_end/bert_loader/query_index.py - SST/Yelp Query Index ---====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#
#
# Purpose:
#   Reads one row of an SST/Yelp query-index CSV and builds the wrapped
#   compact BERT model of that query through the bert creator path, so the
#   backend CLI can verify the query without a per-query ACT JSON file.
#
#===---------------------------------------------------------------------===#

"""Query-index rows for checkpoint-backed SST/Yelp verification queries.

One query is one example, one WordPiece position and one lp ball around that
position's embedding sum (threat model D1, p in {inf, 1}). A query-index CSV
has the columns ``dataset, split, depth, example_id, position, p, eps`` and
an optional ``checkpoint_sha256``; extra columns are ignored. ``example_id``
indexes ``load_bert_dataset(dataset, split)``; a query id is the 0-based
data-row number of the CSV.
"""

from __future__ import annotations

import csv
import hashlib
import logging
import math
import random
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn

from act.front_end.bert_loader.compact_bert import load_compact_bert, resolve_checkpoint
from act.front_end.bert_loader.create_specs import BertSpecCreator
from act.front_end.bert_loader.data_loader import (
    BertExample,
    _synthetic_examples,
    find_bert_dataset_name,
    load_bert_dataset,
)
from act.front_end.model_synthesis import synthesize_models_from_specs
from act.front_end.spec_creator_base import LabeledInputTensor
from act.util.device_manager import get_current_settings
from act.util.path_config import get_data_root

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS: tuple[str, ...] = (
    "dataset",
    "split",
    "depth",
    "example_id",
    "position",
    "p",
    "eps",
)
CHECKSUM_COLUMN = "checkpoint_sha256"
# Model name under which BertSpecCreator creates checkpoint-backed specs.
COMPACT_BERT_MODEL_NAME = "compact_bert"
# Same WordPiece length cap (incl. [CLS]/[SEP]) as BertSpecCreator's default.
MAX_VERIFY_LENGTH = 20
YELP_PARTITION_SEED = 20260928
YELP_PARTITION_SPLITS: tuple[str, ...] = ("calibration", "evaluation")
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


class QueryIndexError(ValueError):
    """A query-index file, row, or the query it names is invalid."""


@dataclass(frozen=True)
class TextQuery:
    """One SST/Yelp verification query parsed from a query-index row."""

    dataset: str
    split: str
    depth: int
    example_id: int
    position: int
    p: float
    eps: float
    checkpoint_sha256: str | None = None

    @property
    def model_dir(self) -> Path:
        """Compact BERT model directory (holding the ``checkpoint`` file) of this query."""
        return (
            Path(get_data_root())
            / "buffet"
            / "models"
            / f"model_{self.dataset}_{self.depth}"
        )


def _field(row: Mapping[str, str | None], column: str, where: str) -> str:
    value = (row.get(column) or "").strip()
    if not value:
        raise QueryIndexError(f"{where}: column '{column}' is empty")
    return value


def _int_field(row: Mapping[str, str | None], column: str, minimum: int, where: str) -> int:
    raw = _field(row, column, where)
    try:
        value = int(raw)
    except ValueError as error:
        raise QueryIndexError(f"{where}: {column}={raw!r} is not an integer") from error
    if value < minimum:
        raise QueryIndexError(f"{where}: {column}={value} must be >= {minimum}")
    return value


def _float_field(row: Mapping[str, str | None], column: str, where: str) -> float:
    raw = _field(row, column, where)
    try:
        return float(raw)
    except ValueError as error:
        raise QueryIndexError(f"{where}: {column}={raw!r} is not a number") from error


def parse_query_row(row: Mapping[str, str | None], where: str = "query") -> TextQuery:
    """Validate one CSV row and return its query.

    Args:
        row: Column name to raw cell text (as produced by ``csv.DictReader``).
        where: Location prefix for error messages.

    Raises:
        QueryIndexError: If a required value is missing or out of range.
    """
    dataset_name = _field(row, "dataset", where)
    try:
        dataset = find_bert_dataset_name(dataset_name)
    except ValueError as error:
        raise QueryIndexError(f"{where}: {error}") from error
    p = _float_field(row, "p", where)
    if not (p == 1.0 or p == math.inf):
        raise QueryIndexError(f"{where}: p={p} is not supported; use inf or 1")
    eps = _float_field(row, "eps", where)
    if not (math.isfinite(eps) and eps > 0.0):
        raise QueryIndexError(f"{where}: eps={eps} must be finite and > 0")
    checksum = (row.get(CHECKSUM_COLUMN) or "").strip().lower() or None
    if checksum is not None and not _SHA256_PATTERN.fullmatch(checksum):
        raise QueryIndexError(f"{where}: {CHECKSUM_COLUMN} is not a hex SHA-256 digest")
    return TextQuery(
        dataset=dataset,
        split=_field(row, "split", where),
        depth=_int_field(row, "depth", 1, where),
        example_id=_int_field(row, "example_id", 0, where),
        position=_int_field(row, "position", 1, where),
        p=p,
        eps=eps,
        checkpoint_sha256=checksum,
    )


def read_query(index_path: str | Path, query_id: int) -> TextQuery:
    """Read and validate data row ``query_id`` (0-based) of a query-index CSV.

    Raises:
        QueryIndexError: If the file lacks a required column, the id is out
            of range, or the row is invalid.
    """
    path = Path(index_path)
    if query_id < 0:
        raise QueryIndexError(f"{path}: query id {query_id} must be >= 0")
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = [name for name in REQUIRED_COLUMNS if name not in (reader.fieldnames or [])]
        if missing:
            raise QueryIndexError(f"{path}: missing required columns {missing}")
        rows = 0
        for row_number, row in enumerate(reader):
            if row_number == query_id:
                return parse_query_row(row, where=f"{path} query {query_id}")
            rows += 1
    raise QueryIndexError(f"{path}: query id {query_id} out of range ({rows} data rows)")


def _recorded_sha256(relative_path: str) -> str:
    """Return the ``data/buffet/SHA256SUMS`` digest of a data-root-relative path."""
    sums_path = Path(get_data_root()) / "buffet" / "SHA256SUMS"
    with open(sums_path, encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2 and parts[1].lstrip("*") == relative_path:
                return parts[0].lower()
    raise QueryIndexError(f"{sums_path} has no entry for {relative_path}")


def verify_checkpoint_sha256(query: TextQuery) -> Path:
    """Check the query's checkpoint against the row and ``SHA256SUMS``.

    The resolved ``ckpt-N/pytorch_model.bin`` must hash to the digest recorded
    in ``data/buffet/SHA256SUMS``, and the row's ``checkpoint_sha256`` must
    equal that digest.

    Returns:
        The verified checkpoint file.

    Raises:
        QueryIndexError: On a missing digest or any mismatch.
    """
    if query.checkpoint_sha256 is None:
        raise QueryIndexError("query has no checkpoint_sha256 to verify")
    checkpoint_file = resolve_checkpoint(query.model_dir) / "pytorch_model.bin"
    relative = (
        f"buffet/models/{query.model_dir.name}/"
        f"{checkpoint_file.parent.name}/{checkpoint_file.name}"
    )
    recorded = _recorded_sha256(relative)
    if query.checkpoint_sha256 != recorded:
        raise QueryIndexError(
            f"checkpoint_sha256 {query.checkpoint_sha256} != SHA256SUMS {recorded} for {relative}"
        )
    digest = hashlib.sha256()
    with open(checkpoint_file, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    if digest.hexdigest() != recorded:
        raise QueryIndexError(f"{checkpoint_file} does not match SHA256SUMS {recorded}")
    return checkpoint_file


def yelp_partition_indices(num_examples: int) -> dict[str, tuple[int, ...]]:
    """Return the fixed 20/80 source-row partition of Yelp ``test.csv``."""
    if num_examples < 0:
        raise ValueError("num_examples must be >= 0")
    shuffled = list(range(num_examples))
    random.Random(YELP_PARTITION_SEED).shuffle(shuffled)
    calibration_size = num_examples // 5
    return {
        "calibration": tuple(sorted(shuffled[:calibration_size])),
        "evaluation": tuple(sorted(shuffled[calibration_size:])),
    }


def _load_query_examples(query: TextQuery) -> list[BertExample]:
    """Load the raw split and enforce Yelp calibration/evaluation membership."""
    partitioned_yelp = (
        query.dataset == "yelp" and query.split in YELP_PARTITION_SPLITS
    )
    source_split = "test" if partitioned_yelp else query.split
    examples = load_bert_dataset(query.dataset, source_split)
    if examples == _synthetic_examples(query.dataset):
        raise QueryIndexError(f"no raw {query.dataset} data for split '{query.split}'")
    if query.example_id >= len(examples):
        raise QueryIndexError(
            f"example_id {query.example_id} out of range for {query.dataset} "
            f"{query.split} ({len(examples)} examples)"
        )
    if partitioned_yelp:
        allowed = yelp_partition_indices(len(examples))[query.split]
        if query.example_id not in allowed:
            raise QueryIndexError(
                f"example_id {query.example_id} is not in Yelp {query.split} split"
            )
    return examples


def build_query_model(query: TextQuery) -> nn.Module:
    """Build the wrapped model (``VerifiableModel``) of one query.

    Follows the ``compact_bert`` path of ``BertSpecCreator``: the
    checkpoint is loaded at the device manager's device/dtype, the example must
    be correctly classified with at most ``MAX_VERIFY_LENGTH`` WordPieces, and
    the spec is the creator's sweep spec for ``query.position``.

    Raises:
        QueryIndexError: If the model, data split, example or position is invalid.
    """
    model_dir = query.model_dir
    if not model_dir.is_dir():
        raise QueryIndexError(f"no compact BERT model directory {model_dir}")
    if query.checkpoint_sha256 is not None:
        verify_checkpoint_sha256(query)
    examples = _load_query_examples(query)
    device, dtype = get_current_settings()
    loaded = load_compact_bert(model_dir, device=device, dtype=dtype)
    try:
        ((tokenized, embeddings, predicted),) = BertSpecCreator._sample_compact_bert_inputs(
            [examples[query.example_id]],
            loaded,
            num_samples=1,
            max_verify_length=MAX_VERIFY_LENGTH,
        )
    except ValueError as error:
        raise QueryIndexError(
            f"example {query.example_id} is not correctly classified by {model_dir.name} "
            f"within {MAX_VERIFY_LENGTH} WordPieces"
        ) from error
    created = BertSpecCreator(config_dict={})._create_spec_pair(
        embeddings=embeddings,
        predicted=predicted,
        epsilon=query.eps,
        p_norm=query.p,
        perturbed_words=1,
        position_mode="sweep",
        wordpiece_tokens=tokenized.wordpiece_tokens,
    )
    pairs = created if isinstance(created, list) else [created]
    by_position = {
        int(torch.as_tensor(in_spec.perturbed_positions).reshape(-1)[0]): (in_spec, out_spec)
        for in_spec, out_spec in pairs
    }
    if query.position not in by_position:
        raise QueryIndexError(
            f"position {query.position} is not a sweep position of example "
            f"{query.example_id} (L={len(tokenized.wordpiece_tokens)}, "
            f"eligible {sorted(by_position)})"
        )
    labeled = LabeledInputTensor(
        tensor=embeddings,
        label=torch.tensor([predicted], dtype=torch.int64, device=embeddings.device),
    )
    wrapped = synthesize_models_from_specs(
        [(query.dataset, COMPACT_BERT_MODEL_NAME, loaded.model, [labeled], [by_position[query.position]])]
    )
    (model,) = wrapped.values()
    logger.info(
        "query %s d%d %s example %d position %d (L=%d, p=%s, eps=%g)",
        query.dataset,
        query.depth,
        query.split,
        query.example_id,
        query.position,
        len(tokenized.wordpiece_tokens),
        query.p,
        query.eps,
    )
    return model
