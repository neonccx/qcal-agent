"""End-to-end single-qubit calibration workflow with quality-gated updates."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from .experiments import EXPERIMENTS, ExperimentSuite
from .models import _jsonable


class CalibrationWorkflow:
    def __init__(self, suite: ExperimentSuite) -> None:
        self.suite = suite

    def run(self) -> list:
        results = []
        for name in EXPERIMENTS:
            result = getattr(self.suite, name)()
            results.append(result)
            if result.status == "success":
                self.suite.calibration.update(name, **result.suggested_updates)
        output = Path(self.suite.output_dir)
        calibration = _jsonable(asdict(self.suite.calibration))
        (output / "final_calibration.json").write_text(json.dumps(calibration, ensure_ascii=False, indent=2), encoding="utf-8")
        summary = [{"experiment": r.experiment, "status": r.status, "quality": r.quality, "updates": r.suggested_updates} for r in results]
        (output / "summary.json").write_text(json.dumps(_jsonable(summary), ensure_ascii=False, indent=2), encoding="utf-8")
        return results
