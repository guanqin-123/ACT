#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "onnx"]
# ///
"""Generate deterministic ONNX and VNNLIB conversion coverage fixtures."""

from __future__ import annotations

import hashlib
from importlib import import_module
import json
from pathlib import Path
from typing import Protocol, cast

import numpy as np
import torch
from torch import nn

from act.front_end.spec_creator_base import LabeledInputTensor
from act.front_end.specs import InKind, InputSpec, OutKind, OutputSpec
from act.front_end.verifiable_model import InputLayer, InputSpecLayer, OutputSpecLayer, VerifiableModel
from act.pipeline.verification.act2torch import ACTToTorch
from act.pipeline.verification.torch2act import TorchToACT
from act.util.device_manager import get_default_dtype, initialize_device


class OnnxHelper(Protocol):
    """Typed subset of the ONNX graph-construction module."""

    def make_node(
        self,
        op_type: str,
        inputs: list[str],
        outputs: list[str],
        **attributes: object,
    ) -> object: ...

    def make_tensor_value_info(
        self,
        name: str,
        element_type: int,
        shape: list[int],
    ) -> object: ...

    def make_graph(
        self,
        nodes: list[object],
        name: str,
        inputs: list[object],
        outputs: list[object],
        *,
        initializer: list[object],
    ) -> object: ...

    def make_model(
        self,
        graph: object,
        *,
        opset_imports: list[object],
        ir_version: int,
    ) -> object: ...

    def make_opsetid(self, domain: str, version: int) -> object: ...


class NumpyHelper(Protocol):
    """Typed subset of the ONNX NumPy conversion module."""

    def from_array(self, array: object, name: str) -> object: ...


class OnnxChecker(Protocol):
    """Typed subset of the ONNX checker module."""

    def check_model(self, model: object) -> None: ...


class OnnxWriter(Protocol):
    """Typed subset of the ONNX module used to persist a model."""

    def save_model(self, model: object, destination: Path) -> None: ...


helper = cast(OnnxHelper, cast(object, import_module("onnx.helper")))
numpy_helper = cast(NumpyHelper, cast(object, import_module("onnx.numpy_helper")))
checker = cast(OnnxChecker, cast(object, import_module("onnx.checker")))
onnx_writer = cast(OnnxWriter, cast(object, import_module("onnx")))
ONNX_FLOAT = 1
ONNX_INT64 = 7


class ConversionOps(nn.Module):
    """Native source for the broadcast and attention-operator fixture."""

    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(2, 2)
        self.classifier = nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            self.projection.weight.fill_(0.1)
            self.projection.bias.fill_(0.05)
            self.classifier.weight.copy_(torch.eye(2))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Project tokens, broadcast one row, and reduce pairwise products."""
        projected = self.projection(value)
        row = projected[:, 0].reshape(1, 1, 2)
        mixed = 2.0 * (projected * row) + 1.0
        paired = (mixed + mixed) / 2.0
        scores = torch.matmul(paired, paired.transpose(-1, -2))
        reduced = scores.mean(dim=-1, keepdim=True).flatten(1)
        return self.classifier(reduced)


def check_conversion(model: nn.Module, sample: torch.Tensor) -> None:
    """Check that a generated fixture survives the public conversion round trip."""
    expected = model(sample)
    wrapped = VerifiableModel(
        input_layer=InputLayer(LabeledInputTensor(sample, None), tuple(sample.shape), sample.dtype),
        input_spec=InputSpecLayer(InputSpec(kind=InKind.BOX, lb=sample - 0.1, ub=sample + 0.1)),
        model=model,
        output_spec=OutputSpecLayer(OutputSpec(
            kind=OutKind.RANGE, lb=torch.full_like(expected, -100), ub=torch.full_like(expected, 100),
        )),
    )
    rebuilt = ACTToTorch(TorchToACT(wrapped).run()).run()
    torch.testing.assert_close(
        rebuilt.get_submodule("model")(sample), expected, rtol=1e-9, atol=1e-9,
    )


def write_coverage_ops(fixtures: Path) -> None:
    """Write the compact ONNX/VNNLIB category before it is staged into data."""
    category = fixtures / "coverage_ops"
    (category / "onnx").mkdir(parents=True, exist_ok=True)
    (category / "vnnlib").mkdir(exist_ok=True)
    constants = [
        numpy_helper.from_array(np.array([[[1]], [[-0.5]]], dtype=np.float32), "weight"),
        numpy_helper.from_array(np.array(3, dtype=np.float32), "exponent"),
        numpy_helper.from_array(np.array([[[1], [2]]], dtype=np.float32), "bias"),
        numpy_helper.from_array(np.array(2, dtype=np.float32), "denominator"),
        numpy_helper.from_array(np.array([0], dtype=np.int64), "indices"),
    ]
    nodes = [
        helper.make_node("Conv", ["X", "weight"], ["conv"], kernel_shape=[1]),
        helper.make_node("Pow", ["conv", "exponent"], ["power"]),
        helper.make_node("Add", ["power", "bias"], ["shift"]),
        helper.make_node("Div", ["shift", "denominator"], ["divided"]),
        helper.make_node("Gather", ["divided", "indices"], ["selected"], axis=1),
        helper.make_node("Flatten", ["selected"], ["Y"], axis=1),
    ]
    graph = helper.make_graph(
        nodes,
        "coverage_ops",
        [helper.make_tensor_value_info("X", ONNX_FLOAT, [1, 1, 4])],
        [helper.make_tensor_value_info("Y", ONNX_FLOAT, [1, 4])],
        initializer=constants,
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 13)],
        ir_version=8,
    )
    checker.check_model(model)
    onnx_writer.save_model(model, category / "onnx/operations.onnx")
    constraints = [
        "(vnnlib-version <2.0>)",
        "(declare-network N",
        "  (declare-input X float32 [1,1,4])",
        "  (declare-output Y float32 [1,4]))",
    ]
    for index in range(4):
        constraints.extend(
            [
                f"(assert (>= X[0,0,{index}] -1))",
                f"(assert (<= X[0,0,{index}] 1))",
            ]
        )
    bounded = "\n".join([*constraints, "(assert (>= Y[0,0] 100))"]) + "\n"
    disjunction = (
        "\n".join(constraints)
        + "\n(assert (or (>= Y[0,0] 100) (<= Y[0,1] -100)))\n"
    )
    _ = (category / "vnnlib/bounded.vnnlib").write_text(bounded, encoding="utf-8")
    _ = (category / "vnnlib/disjunction.vnnlib").write_text(disjunction, encoding="utf-8")
    _ = (category / "instances.csv").write_text(
        "onnx/operations.onnx,vnnlib/bounded.vnnlib,60\n"
        "onnx/native_ops.onnx,vnnlib/native_ops.vnnlib,60\n",
        encoding="utf-8",
    )
    source = ConversionOps().eval()
    check_conversion(source, torch.linspace(-0.1, 0.1, 4, dtype=get_default_dtype()).reshape(1, 2, 2))
    native_constants = [
        numpy_helper.from_array(source.projection.weight.detach().numpy().T.copy(), "W"),
        numpy_helper.from_array(source.projection.bias.detach().numpy(), "B"),
        numpy_helper.from_array(np.array(0, dtype=np.int64), "row_index"),
        numpy_helper.from_array(np.array([1, 1, 2], dtype=np.int64), "row_shape"),
        numpy_helper.from_array(np.array(2, dtype=np.float64), "two"),
        numpy_helper.from_array(np.array(1, dtype=np.float64), "one"),
    ]
    native_nodes = [
        helper.make_node("MatMul", ["X", "W"], ["linear"]),
        helper.make_node("Add", ["linear", "B"], ["projected"]),
        helper.make_node("Gather", ["projected", "row_index"], ["row"], axis=1),
        helper.make_node("Reshape", ["row", "row_shape"], ["row_view"]),
        helper.make_node("Mul", ["projected", "row_view"], ["product"]),
        helper.make_node("Mul", ["two", "product"], ["scaled"]),
        helper.make_node("Add", ["scaled", "one"], ["mixed"]),
        helper.make_node("Add", ["mixed", "mixed"], ["sum"]),
        helper.make_node("Div", ["sum", "two"], ["paired"]),
        helper.make_node("Transpose", ["paired"], ["transposed"], perm=[0, 2, 1]),
        helper.make_node("MatMul", ["paired", "transposed"], ["scores"]),
        helper.make_node("ReduceMean", ["scores"], ["reduced"], axes=[-1], keepdims=1),
        helper.make_node("Flatten", ["reduced"], ["Y"], axis=1),
    ]
    native_graph = helper.make_graph(
        native_nodes, "native_ops",
        [helper.make_tensor_value_info("X", 11, [1, 2, 2])],
        [helper.make_tensor_value_info("Y", 11, [1, 2])], initializer=native_constants,
    )
    native_model = helper.make_model(
        native_graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=8,
    )
    checker.check_model(native_model)
    onnx_writer.save_model(native_model, category / "onnx/native_ops.onnx")
    native_bounds = [
        f"(assert ({op} X[0,{i // 2},{i % 2}] {bound}))"
        for i in range(4) for op, bound in ((">=", -0.1), ("<=", 0.1))
    ]
    _ = (category / "vnnlib/native_ops.vnnlib").write_text(
        "(vnnlib-version <2.0>)\n(declare-network N\n"
        "  (declare-input X float64 [1,2,2])\n  (declare-output Y float64 [1,2]))\n"
        + "\n".join([*native_bounds, "(assert (>= Y[0,0] 100))"]) + "\n",
        encoding="utf-8",
    )


def print_manifest(fixtures: Path) -> None:
    """Print reproducibility hashes for all generated conversion fixtures."""
    paths = [
        *sorted((fixtures / "coverage_ops").rglob("*")),
    ]
    manifest = [
        {
            "path": str(path.relative_to(fixtures)),
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in paths
        if path.is_file()
    ]
    print(json.dumps(manifest, indent=2))


def main() -> None:
    """Generate the ONNX/VNNLIB category beneath this script's directory."""
    initialize_device("cpu", "float64")
    fixtures = Path(__file__).parent
    write_coverage_ops(fixtures)
    print_manifest(fixtures)


if __name__ == "__main__":
    main()
