from __future__ import annotations

from pathlib import Path
from typing import TypedDict, cast

import yaml

from act.back_end.config import BaBConfig


class BabSection(TypedDict):
    alpha_split_objective: bool
    alpha_iters: int
    lr_alpha: float
    lambda_intermediate: float


class BackendSection(TypedDict):
    bab: BabSection


class ConfigDoc(TypedDict):
    backend: BackendSection


def test_bab_config_alpha_split_default_off() -> None:
    assert BaBConfig().alpha_split_objective is False


def test_alpha_split_yaml_override() -> None:
    config_path = Path(__file__).with_name("config.yaml")
    with config_path.open(encoding="utf-8") as handle:
        config_doc = cast(ConfigDoc, yaml.safe_load(handle) or {})

    bab = config_doc["backend"]["bab"]
    assert bab["alpha_split_objective"] is False
    assert bab["alpha_iters"] == 10
    assert bab["lr_alpha"] == 0.5
    assert bab["lambda_intermediate"] == 1.0


def test_lr_alpha_default_matches_abcrown() -> None:
    assert BaBConfig().lr_alpha == 0.5


def test_lambda_intermediate_default_is_one() -> None:
    assert BaBConfig().lambda_intermediate == 1.0
