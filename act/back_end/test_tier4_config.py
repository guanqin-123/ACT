from __future__ import annotations

from pathlib import Path
from typing import TypedDict, cast

import yaml

from act.back_end.config import BaBConfig


class BabSection(TypedDict):
    alpha_per_spec: bool


class BackendSection(TypedDict):
    bab: BabSection


class ConfigDoc(TypedDict):
    backend: BackendSection


def test_bab_config_alpha_per_spec_default_off() -> None:
    assert BaBConfig().alpha_per_spec is False


def test_alpha_per_spec_yaml_roundtrip(tmp_path: Path) -> None:
    config_path = Path(__file__).with_name("config.yaml")
    with config_path.open(encoding="utf-8") as handle:
        config_doc = cast(ConfigDoc, yaml.safe_load(handle) or {})
    assert config_doc["backend"]["bab"]["alpha_per_spec"] is False

    cfg_off = BaBConfig.from_yaml(config_path)
    assert cfg_off.alpha_per_spec is False

    cfg_on = BaBConfig.from_yaml(config_path, alpha_per_spec=True)
    assert cfg_on.alpha_per_spec is True

    out_path = tmp_path / "tier4.yaml"
    cfg_on.to_yaml(out_path)
    cfg_reloaded = BaBConfig.from_yaml(out_path)
    assert cfg_reloaded.alpha_per_spec is True
