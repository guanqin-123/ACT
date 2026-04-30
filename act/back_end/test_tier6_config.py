from __future__ import annotations

from pathlib import Path
from typing import TypedDict, cast

import yaml

from act.back_end.config import BaBConfig


class BabSection(TypedDict):
    eta_per_spec: bool


class BackendSection(TypedDict):
    bab: BabSection


class ConfigDoc(TypedDict):
    backend: BackendSection


def test_bab_config_eta_per_spec_default_off() -> None:
    assert BaBConfig().eta_per_spec is False


def test_eta_per_spec_yaml_roundtrip(tmp_path: Path) -> None:
    config_path = Path(__file__).with_name("config.yaml")
    with config_path.open(encoding="utf-8") as handle:
        config_doc = cast(ConfigDoc, yaml.safe_load(handle) or {})
    assert config_doc["backend"]["bab"]["eta_per_spec"] is False

    cfg_off = BaBConfig.from_yaml(config_path)
    assert cfg_off.eta_per_spec is False

    cfg_on = BaBConfig.from_yaml(config_path, eta_per_spec=True)
    assert cfg_on.eta_per_spec is True

    out_path = tmp_path / "tier6.yaml"
    cfg_on.to_yaml(out_path)
    cfg_reloaded = BaBConfig.from_yaml(out_path)
    assert cfg_reloaded.eta_per_spec is True


def test_eta_per_spec_cli_flag_threads_to_bab_config() -> None:
    from scripts.verify_resnet_cifar100 import _build_config, _parse_args, _to_cli_args

    parsed_off = _parse_args(["--instance", "3995"])
    cli_off = _to_cli_args(parsed_off)
    assert cli_off.eta_per_spec is False

    parsed_on = _parse_args(["--instance", "3995", "--eta-per-spec"])
    cli_on = _to_cli_args(parsed_on)
    assert cli_on.eta_per_spec is True

    cfg_on = _build_config(
        subproblem_batch_size=1,
        eta_iters=1,
        max_depth=1,
        max_nodes=1,
        verbose=False,
        record_bound_trace=False,
        alpha_split=False,
        lambda_intermediate=0.0,
        alpha_chunk_size=4096,
        lambda_intermediate_max_width=None,
        alpha_iters=1,
        lr_alpha=0.5,
        alpha_per_spec=False,
        eta_per_spec=True,
    )
    assert cfg_on.eta_per_spec is True

    cfg_off = _build_config(
        subproblem_batch_size=1,
        eta_iters=1,
        max_depth=1,
        max_nodes=1,
        verbose=False,
        record_bound_trace=False,
        alpha_split=False,
        lambda_intermediate=0.0,
        alpha_chunk_size=4096,
        lambda_intermediate_max_width=None,
        alpha_iters=1,
        lr_alpha=0.5,
        alpha_per_spec=False,
        eta_per_spec=False,
    )
    assert cfg_off.eta_per_spec is False
