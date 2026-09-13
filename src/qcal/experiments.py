"""Runnable experiment suite: acquire, analyze, plot, and persist."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np

from . import analysis
from .backend import MeasurementBackend
from .devices import QubitCalibration
from .models import ExperimentResult
from .plotting import plot_result


class ExperimentSuite:
    def __init__(self, backend: MeasurementBackend, calibration: QubitCalibration, output_dir: str | Path) -> None:
        self.backend = backend
        self.calibration = calibration
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _run(self, name: str, analyzer: Callable, **parameters) -> ExperimentResult:
        raw = self.backend.acquire(name, self.calibration.qubit_id, **parameters)
        if name == "iq_raw":
            fitted, quality, updates = analyzer(raw, self.calibration.qubit_frequency_hz)
        elif name == "ramsey":
            fitted, quality, updates = analyzer(raw, parameters["drive_frequency_hz"])
        else:
            fitted, quality, updates = analyzer(raw)
        status = "success" if quality.get("fit_ok", True) else "warning"
        message = "analysis passed quality checks" if status == "success" else "result saved; inspect quality metrics before applying"
        result = ExperimentResult(name, self.calibration.qubit_id, parameters, raw, fitted, quality, updates, status=status, message=message)
        figure = plot_result(name, raw, fitted, self.output_dir / f"{name}.png")
        result.figures.append(figure)
        result.save_json(self.output_dir / f"{name}.json")
        return result

    def resonator_spectroscopy(self, frequencies_hz=None) -> ExperimentResult:
        center = self.calibration.resonator_frequency_hz
        frequencies_hz = np.linspace(center - 45e6, center + 45e6, 401) if frequencies_hz is None else frequencies_hz
        return self._run("resonator_spectroscopy", analysis.analyze_resonator_spectroscopy, frequencies_hz=frequencies_hz)

    def resonator_punchout(self, frequencies_hz=None, powers_dbm=None) -> ExperimentResult:
        center = self.calibration.resonator_frequency_hz
        frequencies_hz = np.linspace(center - 22e6, center + 22e6, 281) if frequencies_hz is None else frequencies_hz
        powers_dbm = np.linspace(-35, -5, 31) if powers_dbm is None else powers_dbm
        return self._run("resonator_punchout", analysis.analyze_resonator_punchout, frequencies_hz=frequencies_hz, powers_dbm=powers_dbm)

    def resonator_flux(self, frequencies_hz=None, flux_bias=None) -> ExperimentResult:
        center = self.calibration.resonator_frequency_hz
        frequencies_hz = np.linspace(center - 18e6, center + 22e6, 301) if frequencies_hz is None else frequencies_hz
        flux_bias = np.linspace(-.25, .25, 61) if flux_bias is None else flux_bias
        return self._run("resonator_flux", analysis.analyze_resonator_flux, frequencies_hz=frequencies_hz, flux_bias=flux_bias)

    def qubit_spectroscopy(self, frequencies_hz=None) -> ExperimentResult:
        center = self.calibration.qubit_frequency_hz
        frequencies_hz = np.linspace(center - 140e6, center + 140e6, 501) if frequencies_hz is None else frequencies_hz
        return self._run("qubit_spectroscopy", analysis.analyze_qubit_spectroscopy, frequencies_hz=frequencies_hz)

    def iq_raw(self, shots: int = 5000) -> ExperimentResult:
        return self._run("iq_raw", analysis.analyze_iq_raw, shots=shots)

    def rabi(self, amplitudes=None) -> ExperimentResult:
        amplitudes = np.linspace(0, 1.05, 161) if amplitudes is None else amplitudes
        return self._run("rabi", analysis.analyze_rabi, amplitudes=amplitudes)

    def ramsey(self, delays_s=None, drive_frequency_hz: float | None = None) -> ExperimentResult:
        delays_s = np.linspace(0, 80e-6, 241) if delays_s is None else delays_s
        drive = self.calibration.qubit_frequency_hz + 1.2e6 if drive_frequency_hz is None else drive_frequency_hz
        return self._run("ramsey", analysis.analyze_ramsey, delays_s=delays_s, drive_frequency_hz=drive)

    def t1(self, delays_s=None) -> ExperimentResult:
        delays_s = np.linspace(0, 180e-6, 181) if delays_s is None else delays_s
        return self._run("t1", analysis.analyze_t1, delays_s=delays_s)

    def drag(self, betas=None, repetitions: int = 20) -> ExperimentResult:
        betas = np.linspace(-.9, .6, 121) if betas is None else betas
        return self._run("drag", analysis.analyze_drag, betas=betas, repetitions=repetitions)

    def single_qubit_xeb(self, depths=None, circuits: int = 100, shots: int = 1000) -> ExperimentResult:
        depths = np.array([1, 2, 4, 8, 12, 16, 24, 32, 48]) if depths is None else depths
        return self._run("single_qubit_xeb", analysis.analyze_xeb, depths=depths, circuits=circuits, shots=shots)

    def single_qubit_rb(self, lengths=None, sequences: int = 50, shots: int = 1000) -> ExperimentResult:
        lengths = np.array([1, 2, 4, 8, 12, 16, 24, 32, 48]) if lengths is None else lengths
        return self._run("single_qubit_rb", analysis.analyze_rb, lengths=lengths, sequences=sequences, shots=shots)


EXPERIMENTS = (
    "resonator_spectroscopy", "resonator_punchout", "resonator_flux", "qubit_spectroscopy",
    "iq_raw", "rabi", "ramsey", "t1", "drag", "single_qubit_xeb", "single_qubit_rb",
)
