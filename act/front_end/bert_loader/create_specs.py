#===- act/front_end/bert_loader/create_specs.py - BERT Specs ------------====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#
#
# Purpose:
#   Creates embedding-space InputSpec and OutputSpec pairs for SST/Yelp BERT
#   classifiers whose verification graph starts after token embedding lookup.
#
#===---------------------------------------------------------------------===#

"""Specification creator for BERT verify-from-embeddings tasks."""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, cast

import torch
import torch.nn as nn

from act.front_end.spec_creator_base import BaseSpecCreator, LabeledInputTensor
from act.front_end.specs import InKind, InputSpec, OutKind, OutputSpec
from act.front_end.bert_loader.data_loader import (
    BertEmbeddingClassifier,
    BertVocabulary,
    find_bert_dataset_name,
    load_bert_dataset,
    sample_correctly_classified,
)
from act.front_end.bert_loader.compact_bert import (
    CompactBertTokenizedInput,
    LoadedCompactBert,
    embedding_sum,
    load_compact_bert,
    tokenize_act_tokens,
)
from act.util.device_manager import get_current_settings
from act.util.path_config import get_data_root

logger = logging.getLogger(__name__)


class BertSpecCreator(BaseSpecCreator):
    """Create verification specifications from BERT dataset-model pairs."""

    def __init__(
        self,
        config_name: str | None = None,
        config_dict: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the BERT specification creator.

        Args:
            config_name: Optional YAML config name.
            config_dict: Runtime configuration overrides.
        """
        super().__init__(config_name, config_dict)

    def create_specs_for_data_model_pairs(
        self,
        max_samples: int | None = None,
        filter_fn: Callable[[str, str], bool] | None = None,
        validate_shapes: bool = True,
        *,
        dataset_names: list[str] | None = None,
        model_names: list[str] | None = None,
        num_samples: int = 1,
        split: str = "test",
        max_verify_length: int = 20,
        epsilon: float | None = 0.1,
        p_norm: float | None = 1.0,
        perturbed_words: int | None = 1,
        checkpoint_dir: str | Path | None = None,
        position_mode: str | None = None,
        conversion_route: str | None = None,
    ) -> list[tuple[str, str, nn.Module, list[torch.Tensor], list[tuple[InputSpec, OutputSpec]]]]:
        """Create embedding-space specs for BERT datasets.

        Args:
            dataset_names: Dataset names, defaulting to SST.
            model_names: ``embedding_classifier`` or ``compact_bert``.
            num_samples: Number of correctly classified examples per pair.
            split: Dataset split to read.
            max_verify_length: Maximum token sequence length to verify.
            epsilon: Embedding-space perturbation radius.
            p_norm: Lp norm metadata carried by ``LP_EMBEDDING``.
            perturbed_words: Number of token positions to perturb from the start.
            checkpoint_dir: Compact BERT model directory containing the ``checkpoint`` file.
            position_mode: ``prefix`` for the existing behavior or ``sweep`` for
                one spec per non-continuation WordPiece position.
            conversion_route: ``b`` for generic FX lowering or ``a`` for the
                manual comparison baseline.
            validate_shapes: Whether to validate spec/model shape compatibility.

        Returns:
            List of ``(dataset, model_name, model_from_embeddings, labeled_embeddings, spec_pairs)``.
        """
        if max_samples is not None:
            num_samples = max_samples
        if epsilon is None:
            epsilon = float(self.config.get("eps", 0.1))
        if p_norm is None:
            p_norm = float(self.config.get("p", 1.0))
        if perturbed_words is None:
            perturbed_words = int(self.config.get("perturbed_words", 1))
        if checkpoint_dir is None:
            configured_checkpoint = self.config.get("checkpoint_dir")
            checkpoint_dir = (
                cast(str | Path, configured_checkpoint)
                if configured_checkpoint
                else None
            )
        if position_mode is None:
            position_mode = str(self.config.get("position_mode", "prefix"))
        if position_mode not in {"prefix", "sweep"}:
            raise ValueError("position_mode must be 'prefix' or 'sweep'")
        if conversion_route is None:
            conversion_route = str(self.config.get("conversion_route", "b"))
        if conversion_route not in {"a", "b"}:
            raise ValueError("conversion_route must be 'a' or 'b'")
        datasets = [find_bert_dataset_name(name) for name in (dataset_names or ["sst"])]
        models = model_names or ["embedding_classifier"]
        results: list[
            tuple[str, str, nn.Module, list[LabeledInputTensor], list[tuple[InputSpec, OutputSpec]]]
        ] = []

        for dataset in datasets:
            examples = load_bert_dataset(dataset, split)
            vocabulary = BertVocabulary(
                examples,
                embedding_dim=int(self.config.get("embedding_dim", 8)),
            )
            for model_name in models:
                if filter_fn is not None and not filter_fn(dataset, model_name):
                    continue
                if model_name == "embedding_classifier":
                    model = BertEmbeddingClassifier().eval()
                    selected = sample_correctly_classified(
                        examples,
                        model,
                        vocabulary,
                        num_samples=num_samples,
                        max_verify_length=max_verify_length,
                    )
                    labeled_embeddings = [
                        LabeledInputTensor(
                            tensor=embeddings,
                            label=torch.tensor([predicted], dtype=torch.int64),
                        )
                        for _, embeddings, predicted in selected
                    ]
                    spec_pairs = [
                        cast(
                            tuple[InputSpec, OutputSpec],
                            self._create_spec_pair(
                                embeddings=embeddings,
                                predicted=predicted,
                                epsilon=epsilon,
                                p_norm=p_norm,
                                perturbed_words=perturbed_words,
                            ),
                        )
                        for _, embeddings, predicted in selected
                    ]
                    if validate_shapes:
                        spec_pairs = self._validate_and_filter_specs(
                            spec_pairs, model, labeled_embeddings[0].tensor
                        )
                    if spec_pairs:
                        results.append(
                            (dataset, model_name, model, labeled_embeddings, spec_pairs)
                        )
                    continue

                if model_name != "compact_bert":
                    logger.warning("Skipping unsupported BERT model '%s'", model_name)
                    continue
                if checkpoint_dir is None:
                    raise ValueError(
                        "compact_bert requires checkpoint_dir to name a "
                        "compact BERT model directory"
                    )
                model_dir = Path(checkpoint_dir)
                if not model_dir.is_absolute():
                    model_dir = Path(get_data_root()) / model_dir
                device, dtype = get_current_settings()
                loaded = load_compact_bert(model_dir, device=device, dtype=dtype)
                loaded.model._act_conversion_route = conversion_route
                selected_compact_bert = self._sample_compact_bert_inputs(
                    examples,
                    loaded,
                    num_samples=num_samples,
                    max_verify_length=max_verify_length,
                )
                for tokenized, embeddings, predicted in selected_compact_bert:
                    labeled = LabeledInputTensor(
                        tensor=embeddings,
                        label=torch.tensor(
                            [predicted], dtype=torch.int64, device=embeddings.device
                        ),
                    )
                    created = self._create_spec_pair(
                        embeddings=embeddings,
                        predicted=predicted,
                        epsilon=epsilon,
                        p_norm=p_norm,
                        perturbed_words=perturbed_words,
                        position_mode=position_mode,
                        wordpiece_tokens=tokenized.wordpiece_tokens,
                    )
                    spec_pairs = created if isinstance(created, list) else [created]
                    if validate_shapes:
                        spec_pairs = self._validate_and_filter_specs(
                            spec_pairs, loaded.model, embeddings
                        )
                    if spec_pairs:
                        results.append(
                            (
                                dataset,
                                model_name,
                                loaded.model,
                                [labeled],
                                spec_pairs,
                            )
                        )

        return cast(
            list[tuple[str, str, nn.Module, list[torch.Tensor], list[tuple[InputSpec, OutputSpec]]]],
            results,
        )

    def _create_spec_pair(
        self,
        *,
        embeddings: torch.Tensor,
        predicted: int,
        epsilon: float,
        p_norm: float,
        perturbed_words: int,
        position_mode: str = "prefix",
        wordpiece_tokens: Sequence[str] | None = None,
    ) -> tuple[InputSpec, OutputSpec] | list[tuple[InputSpec, OutputSpec]]:
        """Create a prefix spec or sweep every eligible WordPiece position."""
        length = embeddings.shape[-2]
        if position_mode == "prefix":
            count = max(0, min(perturbed_words, length))
            positions = torch.arange(count, dtype=torch.long, device=embeddings.device)
            return self._single_spec_pair(
                embeddings, predicted, epsilon, p_norm, positions
            )
        if position_mode != "sweep":
            raise ValueError("position_mode must be 'prefix' or 'sweep'")
        if wordpiece_tokens is None or len(wordpiece_tokens) != length:
            raise ValueError(
                "sweep mode requires WordPiece tokens aligned with the embedding sequence"
            )
        return [
            self._single_spec_pair(
                embeddings,
                predicted,
                epsilon,
                p_norm,
                torch.tensor([position], dtype=torch.long, device=embeddings.device),
            )
            for position in range(1, length - 1)
            if not wordpiece_tokens[position].startswith("##")
        ]

    @staticmethod
    def _single_spec_pair(
        embeddings: torch.Tensor,
        predicted: int,
        epsilon: float,
        p_norm: float,
        positions: torch.Tensor,
    ) -> tuple[InputSpec, OutputSpec]:
        """Create one embedding input/output robustness spec pair."""
        input_spec = InputSpec(
            kind=InKind.LP_EMBEDDING,
            center=embeddings.clone(),
            eps=torch.tensor([epsilon], dtype=embeddings.dtype),
            p_norm=p_norm,
            perturbed_positions=positions,
        )
        output_spec = OutputSpec(
            kind=OutKind.MARGIN_ROBUST,
            y_true=torch.tensor([predicted], dtype=torch.int64),
            margin=torch.tensor([0.0], dtype=embeddings.dtype),
        )
        return input_spec, output_spec

    @staticmethod
    def _sample_compact_bert_inputs(
        examples: Sequence[Any],
        loaded: LoadedCompactBert,
        *,
        num_samples: int,
        max_verify_length: int,
    ) -> list[tuple[CompactBertTokenizedInput, torch.Tensor, int]]:
        """Select correctly classified checkpoint-backed examples in file order."""
        selected: list[tuple[CompactBertTokenizedInput, torch.Tensor, int]] = []
        with torch.no_grad():
            for example in examples:
                tokenized = tokenize_act_tokens(example.tokens, loaded.tokenizer)
                if tokenized.input_ids.numel() > max_verify_length:
                    continue
                embeddings = embedding_sum(
                    loaded, tokenized.input_ids, tokenized.token_type_ids
                )
                logits = loaded.model(embeddings)
                predicted = int(logits.argmax(dim=-1).item())
                if predicted != example.label:
                    continue
                selected.append((tokenized, embeddings, predicted))
                if len(selected) >= num_samples:
                    break
        if not selected:
            raise ValueError(
                "No correctly classified compact BERT samples found within max_verify_length"
            )
        return selected
