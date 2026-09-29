#===- act/front_end/vnnlib_loader/vnncomp_instance.py - VNN-COMP Instance --====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#
#
# Purpose:
#   Builds the wrapped models of one VNN-COMP instance (ONNX + VNNLIB) in
#   memory with the model-acquisition steps of vnncomp/act_run_instance.py,
#   so the backend CLI can verify the instance without an ACT JSON file.
#
#===---------------------------------------------------------------------===#

"""In-memory model acquisition for one VNN-COMP (ONNX, VNNLIB) instance.

The steps are those of ``vnncomp/act_run_instance.py``, in the same order:
``create_specs_from_paths`` (ONNX -> torch, input-shape probe, VNNLIB parse
and validation) -> ``merge_split_relus`` (collapse provably affine
DENSE -> ReLU -> DENSE sandwiches) -> ``synthesize_models_from_specs``.

A VNNLIB file can yield several query pairs. Pairs that share the input and
output spec kinds and the output constraint (e.g. several input regions of
one property) are batched into ONE wrapped model with one lane per pair;
pairs with different output constraints (a disjunctive property that is not
recognised as top-1) become separate disjunct models. The instance is unsat
only if every lane of every disjunct is certified.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch.nn as nn

from act.front_end.model_synthesis import merge_split_relus, synthesize_models_from_specs
from act.front_end.vnnlib_loader.create_specs import create_specs_from_paths

logger = logging.getLogger(__name__)


class VnncompInstanceError(ValueError):
    """An ONNX + VNNLIB instance cannot be turned into wrapped models."""


def build_instance_models(onnx_path: str | Path, vnnlib_path: str | Path) -> list[nn.Module]:
    """Build the wrapped disjunct models of one VNN-COMP instance.

    Args:
        onnx_path: ONNX network file.
        vnnlib_path: VNNLIB 2.0 property file.

    Returns:
        The wrapped models (``VerifiableModel``) in synthesis order, which is
        the order in which ``act_run_instance.py`` verifies the disjuncts.

    Raises:
        VnncompInstanceError: If a file is missing, the property is invalid or
            unsupported, or no model is synthesized.
    """
    try:
        spec_result = create_specs_from_paths(onnx_path, vnnlib_path)
    except SystemExit as error:
        # create_specs_from_paths reports missing / invalid inputs by exiting;
        # surface them as an ordinary error of this source instead.
        raise VnncompInstanceError(str(error.code)) from error
    category, instance_id, raw_model, labeled_tensors, spec_pairs = spec_result
    # merge_split_relus returns raw_model itself when nothing is merged.
    verify_model, n_merged = merge_split_relus(raw_model)
    if n_merged:
        logger.info("merge_split_relus fused %d split-ReLU neurons", n_merged)
    models = list(
        synthesize_models_from_specs(
            [(category, instance_id, verify_model, labeled_tensors, spec_pairs)]
        ).values()
    )
    if not models:
        raise VnncompInstanceError(
            f"No wrapped models were synthesized for ONNX {str(onnx_path)!r} "
            f"and VNNLIB {str(vnnlib_path)!r}"
        )
    return models
