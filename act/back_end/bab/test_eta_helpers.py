"""Tests for act.back_end.bab.eta helpers."""
import pytest
import torch

from act.back_end.bab.eta import EtaState, expand_eta_state, get_pre_activation_layer_id
from act.back_end.layer_schema import LayerKind


class _SimpleLayer:
    def __init__(self, id: int, kind: str):
        self.id = id
        self.kind = kind


class _SimpleNet:
    def __init__(self, by_id, preds):
        self.by_id = by_id
        self.preds = preds


def _make_net(layers_spec):
    """layers_spec: list of (id, kind_str, pred_ids). Returns a Net with the
    minimum structure required by get_pre_activation_layer_id (just by_id +
    preds). `params` can be empty dicts."""
    by_id = {}
    preds = {}
    for lid, kind, pred_ids in layers_spec:
        L = _SimpleLayer(id=lid, kind=kind)
        by_id[lid] = L
        preds[lid] = list(pred_ids)
    for lid, _k, _p in layers_spec:
        preds.setdefault(lid, [])
    return _SimpleNet(by_id=by_id, preds=preds)


def test_pre_activation_dense_to_relu():
    net = _make_net([
        (0, 'INPUT',       []),
        (1, 'DENSE',       [0]),
        (2, 'RELU',        [1]),
    ])
    assert get_pre_activation_layer_id(net, 2) == 1


def test_pre_activation_conv2d_to_relu():
    net = _make_net([
        (0, 'INPUT',  []),
        (1, 'CONV2D', [0]),
        (2, 'RELU',   [1]),
    ])
    assert get_pre_activation_layer_id(net, 2) == 1


def test_pre_activation_bn_to_relu_maps_to_bn():
    # BN owns the immediate pre-activation.
    net = _make_net([
        (0, 'INPUT', []),
        (1, 'DENSE', [0]),
        (2, 'BN',    [1]),
        (3, 'RELU',  [2]),
    ])
    assert get_pre_activation_layer_id(net, 3) == 2


def test_pre_activation_sigmoid_to_dense():
    net = _make_net([
        (0, 'INPUT',   []),
        (1, 'DENSE',   [0]),
        (2, 'SIGMOID', [1]),
    ])
    assert get_pre_activation_layer_id(net, 2) == 1


def test_pre_activation_rejects_multi_pred():
    # ADD has 2 preds; should reject (unsupported in v1).
    net = _make_net([
        (0, 'INPUT', []),
        (1, 'DENSE', [0]),
        (2, 'DENSE', [0]),
        (3, 'ADD',   [1, 2]),
        (4, 'RELU',  [3]),  # pred is ADD
    ])
    with pytest.raises(ValueError, match=r"(predecessor|not supported|pre-activation kind)"):
        get_pre_activation_layer_id(net, 4)


def test_pre_activation_rejects_non_activation_input():
    net = _make_net([
        (0, 'INPUT', []),
        (1, 'DENSE', [0]),
    ])
    with pytest.raises(ValueError, match=r"activation kind"):
        get_pre_activation_layer_id(net, 1)  # DENSE is not an activation


def test_pre_activation_tanh_to_dense():
    net = _make_net([
        (0, 'INPUT', []),
        (1, 'DENSE', [0]),
        (2, 'TANH',  [1]),
    ])
    assert get_pre_activation_layer_id(net, 2) == 1


def _make_eta(B: int, widths: dict[int, int]) -> EtaState:
    val   = {lid: torch.zeros(B, d) for lid, d in widths.items()}
    sign  = {lid: torch.zeros(B, d) for lid, d in widths.items()}
    point = {lid: torch.zeros(B, d) for lid, d in widths.items()}
    return EtaState(val=val, sign=sign, point=point)


def test_eta_state_empty_and_fast_path():
    eta = EtaState()
    assert eta.is_empty()
    assert eta.fast_path_skip()

    eta2 = _make_eta(B=4, widths={7: 10})
    assert eta2.is_empty()
    eta2.sign[7][2, 3] = 1.0
    assert not eta2.is_empty()


def test_eta_state_to_device_dtype():
    eta = _make_eta(B=2, widths={1: 3, 2: 5})
    eta_d = eta.to(device=torch.device("cpu"), dtype=torch.float64)
    assert eta_d.val[1].dtype == torch.float64
    assert eta_d.val[2].shape == (2, 5)
    assert eta_d.sign[1].device.type == "cpu"


def test_eta_state_select():
    eta = _make_eta(B=5, widths={7: 4})
    for b in range(5):
        eta.sign[7][b, 0] = float(b + 1)
    picked = eta.select(torch.tensor([0, 2, 4], dtype=torch.long))
    assert picked.batch_size == 3
    assert picked.sign[7][:, 0].tolist() == [1.0, 3.0, 5.0]


def test_expand_eta_state_none_passthrough():
    assert expand_eta_state(None, M=3) is None


def test_expand_eta_state_matches_bounds_expansion():
    eta = _make_eta(B=4, widths={2: 6})
    for b in range(4):
        for c in range(6):
            eta.val[2][b, c] = b * 10.0 + c
    M = 3
    big = expand_eta_state(eta, M)
    assert big is not None
    assert big.val[2].shape == (4 * M, 6)
    for b in range(4):
        for j in range(M):
            row = b * M + j
            for c in range(6):
                expected = b * 10.0 + c
                assert big.val[2][row, c].item() == expected, (b, j, c, row)
    same = expand_eta_state(eta, 1)
    assert same is eta


def _make_eta_per_spec(B: int, M: int, widths: dict[int, int]) -> EtaState:
    val = {lid: torch.zeros(B, M, d) for lid, d in widths.items()}
    sign = {lid: torch.zeros(B, d) for lid, d in widths.items()}
    point = {lid: torch.zeros(B, d) for lid, d in widths.items()}
    return EtaState(val=val, sign=sign, point=point, per_spec=True)


def test_eta_state_per_spec_default_off():
    eta = _make_eta(B=2, widths={1: 3})
    assert eta.per_spec is False


def test_eta_state_per_spec_on_validates_3d_val():
    eta = _make_eta_per_spec(B=2, M=3, widths={1: 4, 2: 5})
    assert eta.per_spec is True
    assert eta.val[1].shape == (2, 3, 4)
    assert eta.sign[1].shape == (2, 4)
    assert eta.point[1].shape == (2, 4)


def test_eta_state_per_spec_rejects_2d_val_when_on():
    val = {1: torch.zeros(2, 4)}
    sign = {1: torch.zeros(2, 4)}
    point = {1: torch.zeros(2, 4)}
    with pytest.raises(ValueError, match=r"per_spec=True.*val must be at least 3-D"):
        EtaState(val=val, sign=sign, point=point, per_spec=True)


def test_eta_state_per_spec_rejects_d_dim_mismatch_when_on():
    val = {1: torch.zeros(2, 3, 4)}
    sign = {1: torch.zeros(2, 5)}
    point = {1: torch.zeros(2, 5)}
    with pytest.raises(ValueError, match=r"per_spec=True.*D-dim mismatch"):
        EtaState(val=val, sign=sign, point=point, per_spec=True)


def test_eta_state_per_spec_to_propagates_flag():
    eta = _make_eta_per_spec(B=2, M=3, widths={1: 4})
    moved = eta.to(device=torch.device("cpu"), dtype=torch.float64)
    assert moved.per_spec is True
    assert moved.val[1].dtype == torch.float64


def test_eta_state_per_spec_select_propagates_flag_and_indexes_batch():
    eta = _make_eta_per_spec(B=5, M=3, widths={7: 4})
    for b in range(5):
        eta.val[7][b, :, 0] = float(b + 1)
    picked = eta.select(torch.tensor([0, 2, 4], dtype=torch.long))
    assert picked.per_spec is True
    assert picked.val[7].shape == (3, 3, 4)
    assert picked.sign[7].shape == (3, 4)
    assert picked.val[7][:, 0, 0].tolist() == [1.0, 3.0, 5.0]
