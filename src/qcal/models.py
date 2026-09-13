"""Serializable experiment result contract."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        if np.iscomplexobj(value):
            return {"real": value.real.tolist(), "imag": value.imag.tolist()}
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, complex):
        return {"real": value.real, "imag": value.imag}
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


@dataclass(slots=True)
class ExperimentResult:
    experiment: str
    target: str
    parameters: dict[str, Any]
    raw_data: dict[str, Any]
    fit: dict[str, Any] = field(default_factory=dict)
    quality: dict[str, Any] = field(default_factory=dict)
    suggested_updates: dict[str, float] = field(default_factory=dict)
    figures: list[str] = field(default_factory=list)
    status: str = "success"
    message: str = ""
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(
            {
                "experiment": self.experiment,
                "target": self.target,
                "parameters": self.parameters,
                "raw_data": self.raw_data,
                "fit": self.fit,
                "quality": self.quality,
                "suggested_updates": self.suggested_updates,
                "figures": self.figures,
                "status": self.status,
                "message": self.message,
                "created_at": self.created_at,
            }
        )

    def save_json(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return output
