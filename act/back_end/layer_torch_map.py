#!/usr/bin/env python3
# ===- act/back_end/layer_torch_map.py - ACT/Torch layer mapping ---------====#

"""Canonical LayerKind <-> torch.nn.Module correspondence."""

from typing import override

import torch
import torch.nn as nn

from act.back_end.layer_schema import LayerKind


class _ErfModule(nn.Module):
    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.erf(x)


class _SqrtModule(nn.Module):
    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(torch.clamp(x, min=0.0))


class _QuantizeModule(nn.Module):
    def __init__(self, scale: object = None, zero_point: object = None, qmin: int = 0, qmax: int = 255) -> None:
        super().__init__()
        self.register_buffer("scale", torch.as_tensor(1.0 if scale is None else scale))
        self.register_buffer("zero_point", torch.as_tensor(0 if zero_point is None else zero_point))
        self.qmin: float = float(qmin)
        self.qmax: float = float(qmax)

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.get_buffer("scale").to(device=x.device, dtype=x.dtype)
        zp = self.get_buffer("zero_point").to(device=x.device, dtype=x.dtype)
        return scale * torch.clamp(torch.round(x / scale), min=self.qmin - zp, max=self.qmax - zp)


class _SinModule(nn.Module):
    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(x)


class _CosModule(nn.Module):
    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cos(x)


# ACT LayerKind -> PyTorch nn.Module path.
# Layers not listed are skipped during restoration (wrapper, graph ops, functional-only).
ACT_TO_TORCH: dict[str, type[nn.Module]] = {
    LayerKind.DENSE.value: nn.Linear,
    LayerKind.CONV1D.value: nn.Conv1d,
    LayerKind.CONV2D.value: nn.Conv2d,
    LayerKind.CONV3D.value: nn.Conv3d,
    LayerKind.CONVTRANSPOSE2D.value: nn.ConvTranspose2d,
    LayerKind.MAXPOOL1D.value: nn.MaxPool1d,
    LayerKind.MAXPOOL2D.value: nn.MaxPool2d,
    LayerKind.MAXPOOL3D.value: nn.MaxPool3d,
    LayerKind.AVGPOOL1D.value: nn.AvgPool1d,
    LayerKind.AVGPOOL2D.value: nn.AvgPool2d,
    LayerKind.AVGPOOL3D.value: nn.AvgPool3d,
    LayerKind.ADAPTIVEAVGPOOL2D.value: nn.AdaptiveAvgPool2d,
    LayerKind.RELU.value: nn.ReLU,
    LayerKind.LRELU.value: nn.LeakyReLU,
    LayerKind.PRELU.value: nn.PReLU,
    LayerKind.SIGMOID.value: nn.Sigmoid,
    LayerKind.TANH.value: nn.Tanh,
    LayerKind.ERF.value: _ErfModule,
    LayerKind.SQRT.value: _SqrtModule,
    LayerKind.SIN.value: _SinModule,
    LayerKind.COS.value: _CosModule,
    LayerKind.QUANTIZE.value: _QuantizeModule,
    LayerKind.SOFTPLUS.value: nn.Softplus,
    LayerKind.SILU.value: nn.SiLU,
    LayerKind.GELU.value: nn.GELU,
    LayerKind.RELU6.value: nn.ReLU6,
    LayerKind.HARDTANH.value: nn.Hardtanh,
    LayerKind.HARDSIGMOID.value: nn.Hardsigmoid,
    LayerKind.HARDSWISH.value: nn.Hardswish,
    LayerKind.MISH.value: nn.Mish,
    LayerKind.SOFTSIGN.value: nn.Softsign,
    LayerKind.FLATTEN.value: nn.Flatten,
    LayerKind.EMBEDDING.value: nn.Embedding,
    LayerKind.RNN.value: nn.RNN,
    LayerKind.GRU.value: nn.GRU,
    LayerKind.LSTM.value: nn.LSTM,
    LayerKind.SOFTMAX.value: nn.Softmax,
    LayerKind.MHA.value: nn.MultiheadAttention,
}


__all__ = ["ACT_TO_TORCH"]
