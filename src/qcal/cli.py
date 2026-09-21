"""Command-line entry point for simulated calibration runs."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from .devices import QubitCalibration
from .experiments import EXPERIMENTS, ExperimentSuite
from .simulator import SimulatedBackend
from .workflow import CalibrationWorkflow


def _output_path(value: str | None) -> Path:
    return Path(value) if value else Path("runs") / datetime.now().strftime("%Y%m%d-%H%M%S")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Clean-room single-qubit calibration system")
    sub = parser.add_subparsers(dest="command", required=True)
    full = sub.add_parser("full", help="run the complete calibration workflow")
    full.add_argument("--output", help="output directory")
    full.add_argument("--seed", type=int, default=20260913)
    one = sub.add_parser("experiment", help="run one experiment")
    one.add_argument("name", choices=EXPERIMENTS)
    one.add_argument("--output", help="output directory")
    one.add_argument("--seed", type=int, default=20260913)
    sub.add_parser("list", help="list supported experiments")
    args = parser.parse_args(argv)
    if args.command == "list":
        print("\n".join(EXPERIMENTS))
        return 0
    output = _output_path(args.output)
    backend = SimulatedBackend(seed=args.seed)
    calibration = QubitCalibration()
    suite = ExperimentSuite(backend, calibration, output)
    backend.connect()
    try:
        results = CalibrationWorkflow(suite).run() if args.command == "full" else [getattr(suite, args.name)()]
    finally:
        backend.disconnect()
    for result in results:
        print(f"{result.experiment:28s} {result.status:8s} {result.quality}")
    print(f"Artifacts: {output.resolve()}")
    return 0 if all(r.status == "success" for r in results) and (
        args.command != "full" or len(results) == len(EXPERIMENTS)
    ) else 1


if __name__ == "__main__":
    raise SystemExit(main())
