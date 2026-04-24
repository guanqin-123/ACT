from __future__ import annotations

# pyright: reportUnusedFunction=false, reportUnknownMemberType=false

import pytest
import torch

from act.back_end.core import Layer, Net
from act.back_end.layer_schema import LayerKind
from act.util.device_manager import initialize_device


pytestmark = pytest.mark.xfail(reason="Patches mode not implemented yet (Wave 3)")


@pytest.fixture(scope="module", autouse=True)
def _cpu_float32() -> None:
    initialize_device("cpu", "float32")
@pytest.fixture
def relu_conv_net() -> Net:
    """3×3 Conv + ReLU + 3×3 Conv net with random weights."""
    _ = torch.manual_seed(0)
    input_vars = list(range(3 * 8 * 8))
    conv1_out = list(range(1000, 1000 + 8 * 6 * 6))
    relu_out = list(range(2000, 2000 + 8 * 6 * 6))
    conv2_out = list(range(3000, 3000 + 4 * 4 * 4))
    layers = [
        Layer(
            0,
            LayerKind.INPUT.value,
            {"shape": (1, 3, 8, 8), "dtype": "float32", "num_classes": 4, "value_range": (0.0, 1.0)},
            input_vars,
            input_vars,
        ),
        Layer(
            1,
            LayerKind.INPUT_SPEC.value,
            {"kind": "BOX", "lb": torch.zeros((1, 3, 8, 8)), "ub": torch.ones((1, 3, 8, 8))},
            input_vars,
            input_vars,
        ),
        Layer(
            2,
            LayerKind.CONV2D.value,
            {
                "in_channels": 3,
                "out_channels": 8,
                "kernel_size": 3,
                "stride": 1,
                "padding": 0,
                "dilation": 1,
                "groups": 1,
                "input_shape": (1, 3, 8, 8),
                "output_shape": (1, 8, 6, 6),
                "weight": torch.randn(8, 3, 3, 3),
                "bias": torch.randn(8),
            },
            input_vars,
            conv1_out,
        ),
        Layer(3, LayerKind.RELU.value, {}, conv1_out, relu_out),
        Layer(
            4,
            LayerKind.CONV2D.value,
            {
                "in_channels": 8,
                "out_channels": 4,
                "kernel_size": 3,
                "stride": 1,
                "padding": 0,
                "dilation": 1,
                "groups": 1,
                "input_shape": (1, 8, 6, 6),
                "output_shape": (1, 4, 4, 4),
                "weight": torch.randn(4, 8, 3, 3),
                "bias": torch.randn(4),
            },
            relu_out,
            conv2_out,
        ),
        Layer(5, LayerKind.ASSERT.value, {"kind": "RANGE"}, conv2_out, conv2_out),
    ]
    preds = {0: [], 1: [0], 2: [1], 3: [2], 4: [3], 5: [4]}
    succs = {0: [1], 1: [2], 2: [3], 3: [4], 4: [5], 5: []}
    return Net(layers=layers, preds=preds, succs=succs)


CONV_CONFIGS = [
    (1, 0, 3, 8, 3, 8, 8),
    (1, 1, 3, 8, 3, 8, 8),
    (2, 0, 3, 16, 3, 16, 16),
    (2, 1, 3, 16, 3, 16, 16),
    (1, 0, 1, 4, 5, 10, 10),
    (1, 2, 1, 4, 5, 10, 10),
    (1, 1, 8, 8, 3, 4, 4),
    (2, 1, 8, 8, 3, 32, 32),
    (1, 0, 3, 12, 5, 16, 16),
    (2, 2, 3, 12, 5, 32, 32),
]


@pytest.mark.parametrize("stride,padding,in_c,out_c,kernel,H,W", CONV_CONFIGS)
def test_conv_patches_forward_parity(stride: int, padding: int, in_c: int, out_c: int, kernel: int, H: int, W: int) -> None:
    """Forward bounds through Conv in patches mode == matrix mode."""
    del stride, padding, in_c, out_c, kernel, H, W
    pytest.skip("placeholder — patches mode not implemented")


@pytest.mark.parametrize("stride,padding,in_c,out_c,kernel,H,W", CONV_CONFIGS)
def test_conv_patches_backward_parity(stride: int, padding: int, in_c: int, out_c: int, kernel: int, H: int, W: int) -> None:
    """Backward dual bounds via patches match matrix."""
    del stride, padding, in_c, out_c, kernel, H, W
    pytest.skip("placeholder")


def test_patches_to_matrix_roundtrip() -> None:
    """Patches → Matrix → Patches → same tensor."""
    pytest.skip("placeholder")


def test_conv_chain_parity(relu_conv_net: Net) -> None:
    """Conv → BN → ReLU → Conv chain parity."""
    assert relu_conv_net.layers[2].kind == LayerKind.CONV2D.value
    pytest.skip("placeholder")


def test_resnet_basic_block_parity() -> None:
    """ResNet basic block (Conv→BN→ReLU→Conv→BN + skip ADD) parity."""
    pytest.skip("placeholder")


def test_soundness_concrete_samples_matrix() -> None:
    """Concrete sampling: random x in input box, matrix bound contains f(x)."""
    pytest.skip("placeholder — requires sampling infra")


def test_soundness_concrete_samples_patches() -> None:
    """Concrete sampling: random x in input box, patches bound contains f(x)."""
    pytest.skip("placeholder")
