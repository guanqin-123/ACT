"""Front-end configuration loading."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

import yaml

_DEFAULT_YAML = Path(__file__).with_name("config.yaml")


@dataclass
class FrontEndConfig:
    specs: dict[str, dict[str, Any]] = field(default_factory=dict)
    text_verification: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_yaml(
        cls,
        config_path: Optional[Union[str, Path]] = None,
        **overrides: Any,
    ) -> "FrontEndConfig":
        path = Path(config_path) if config_path else _DEFAULT_YAML
        if not path.exists():
            raise FileNotFoundError(f"Front-end config not found: {path}")

        with open(path) as f:
            raw = yaml.safe_load(f) or {}

        specs = deepcopy(raw.get("specs", {}))
        text_verification = deepcopy(raw.get("text_verification", {}))
        text_verification.update(
            {k: v for k, v in overrides.items() if k in text_verification and v is not None}
        )
        return cls(specs=specs, text_verification=text_verification)

    def spec_config(self, name: Optional[str]) -> dict[str, Any]:
        key = name or "default"
        if key not in self.specs:
            raise KeyError(f"Unknown front-end spec config: {key}")
        return deepcopy(self.specs[key])
