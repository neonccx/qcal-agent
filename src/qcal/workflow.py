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
        output = Path(self.suite.output_dir)
        if any(output.iterdir()):
            raise FileExistsError(f"Workflow output directory must be empty: {output}")
        results = []
        for name in EXPERIMENTS:
            result = getattr(self.suite, name)()
            results.append(result)
            if result.status != "success":
                break
            self.suite.calibration.update(name, **result.suggested_updates)
        calibration = _jsonable(asdict(self.suite.calibration))
        (output / "calibration_checkpoint.json").write_text(json.dumps(calibration, ensure_ascii=False, indent=2), encoding="utf-8")
        complete = len(results) == len(EXPERIMENTS) and all(r.status == "success" for r in results)
        if complete:
            (output / "final_calibration.json").write_text(json.dumps(calibration, ensure_ascii=False, indent=2), encoding="utf-8")
        summary = [{"experiment": r.experiment, "status": r.status, "quality": r.quality, "updates": r.suggested_updates} for r in results]
        (output / "summary.json").write_text(json.dumps(_jsonable(summary), ensure_ascii=False, indent=2), encoding="utf-8")
        status = {
            "status": "complete" if complete else "blocked",
            "completed_experiments": len(results),
            "planned_experiments": len(EXPERIMENTS),
            "stopped_at": None if complete else results[-1].experiment,
            "reason": None if complete else results[-1].message,
        }
        (output / "workflow_status.json").write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
        return results
