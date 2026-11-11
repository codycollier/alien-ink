"""Serialization of experiment configuration for local and remote artifacts."""

from dataclasses import asdict
from pathlib import Path
from typing import Any


def serialize_config(obj: Any) -> dict[str, Any]:
    """Flatten a dataclass (or mapping) for logging; stringify Paths recursively."""

    def _convert(value: Any) -> Any:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, dict):
            return {k: _convert(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [_convert(v) for v in value]
        return value

    if hasattr(obj, "__dataclass_fields__"):
        raw = asdict(obj)
    elif isinstance(obj, dict):
        raw = dict(obj)
    else:
        raise TypeError(
            "serialize_config expects a dataclass or dict, "
            f"got {type(obj).__name__}: {obj!r}"
        )
    return {k: _convert(v) for k, v in raw.items()}
