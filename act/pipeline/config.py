from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Optional

import yaml

from act.back_end.config import BaBConfig
from act.pipeline.fuzzing.actfuzzer import FuzzingConfig

_DEFAULT_YAML = Path(__file__).parent / "config.yaml"


@dataclass
class ValidationConfig:
    solvers: list[str]
    tf_modes: list[str]
    samples: int
    per_neuron_topk: int
    bounds_tolerance: str
    batch_sizes: Optional[list[Optional[int]]]


@dataclass
class PipelineConfig:
    fuzzing: FuzzingConfig
    bab: BaBConfig
    validation: ValidationConfig

    @classmethod
    def from_yaml(
        cls,
        config_path: Optional[str | Path] = None,
        **overrides: Any,
    ) -> "PipelineConfig":
        path = Path(config_path) if config_path else _DEFAULT_YAML
        if not path.exists():
            raise FileNotFoundError(
                f"Pipeline config not found: {path}\nExpected: act/pipeline/config.yaml"
            )

        with open(path) as f:
            yaml_data = yaml.safe_load(f) or {}

        fuzz_overrides = _strip_prefixed_overrides(overrides, "fuzz_")
        bab_overrides = _strip_prefixed_overrides(overrides, "bab_")
        val_overrides = _strip_prefixed_overrides(overrides, "val_")

        fuzzing = FuzzingConfig.from_yaml(path, **fuzz_overrides)
        bab_data = ((yaml_data.get("verification") or {}).get("bab") or {})
        validation_data = yaml_data.get("validation") or {}

        bab = BaBConfig(**_merge_dataclass_fields(BaBConfig, bab_data, bab_overrides))
        validation = ValidationConfig(
            **_merge_dataclass_fields(ValidationConfig, validation_data, val_overrides)
        )
        return cls(fuzzing=fuzzing, bab=bab, validation=validation)


def _strip_prefixed_overrides(overrides: dict[str, Any], prefix: str) -> dict[str, Any]:
    return {
        key[len(prefix):]: value
        for key, value in overrides.items()
        if key.startswith(prefix) and value is not None
    }


def _merge_dataclass_fields(
    dataclass_type: type,
    yaml_values: dict[str, Any],
    overrides: dict[str, Any],
) -> dict[str, Any]:
    valid_keys = {field.name for field in fields(dataclass_type)}
    merged = {key: value for key, value in yaml_values.items() if key in valid_keys}
    merged.update({key: value for key, value in overrides.items() if key in valid_keys})
    return merged
