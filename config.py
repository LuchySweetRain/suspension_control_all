from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def deep_update(base: dict, override: dict) -> dict:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    parent_name = data.pop("inherits", None)
    if parent_name:
        parent = load_config(path.parent / parent_name)
        return deep_update(parent, data)
    return data


def as_namespace(data: dict[str, Any]):
    class Namespace:
        pass

    ns = Namespace()
    for key, value in data.items():
        if isinstance(value, dict):
            value = as_namespace(value)
        setattr(ns, key, value)
    return ns

