#===- act/front_end/text_loader/__init__.py - Text Loader Exports -------====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#
#
# Purpose:
#   Exposes text dataset loading and specification creation for embedding-space
#   verification tasks.
#
#===---------------------------------------------------------------------===#

"""Text front-end exports for embedding-space verification."""

from __future__ import annotations

from act.front_end.text_loader.create_specs import TextSpecCreator, create_text_specs
from act.front_end.text_loader.certify import (
    CertifiedRadiusResult,
    RadiusSearchOptions,
    RadiusStep,
    certify_radius,
    configure_bab_for_method,
    falsified_counterexample_violates,
    soundness_sample_certified,
)
from act.front_end.text_loader.data_loader import (
    TextEmbeddingClassifier,
    TextExample,
    TextVocabulary,
    find_text_dataset_name,
    list_text_datasets,
    load_text_dataset,
    sample_correctly_classified,
)

__all__ = [
    "TextEmbeddingClassifier",
    "TextExample",
    "TextSpecCreator",
    "TextVocabulary",
    "CertifiedRadiusResult",
    "RadiusSearchOptions",
    "RadiusStep",
    "certify_radius",
    "configure_bab_for_method",
    "create_text_specs",
    "falsified_counterexample_violates",
    "find_text_dataset_name",
    "list_text_datasets",
    "load_text_dataset",
    "sample_correctly_classified",
    "soundness_sample_certified",
]
