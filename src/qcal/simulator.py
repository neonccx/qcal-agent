"""Physics-informed, deterministic-enough simulated measurement backend."""

from __future__ import annotations

from typing import Any

import numpy as np

from .backend import MeasurementBackend
from .devices import QubitTruth


class SimulatedBackend(MeasurementBackend):
    """Generate realistic calibration data without pretending to be real hardware.

    The model includes readout noise, T1 decay during IQ acquisition, calibration
    drift, finite-shot noise, and configurable measurement noise.
    """

    def __init__(
        self,
        qubits: dict[str, QubitTruth] | None = None,
        seed: int = 20260913,
        noise_scale: float = 1.0,
    ) -> None:
        default = QubitTruth()
        self.qubits = qubits or {default.qubit_id: default}
        self.rng = np.random.default_rng(seed)
        self.noise_scale = float(noise_scale)
        self.connected = False

    def connect(self) -> None:
        self.connected = True

    def disconnect(self) -> None:
        self.connected = False

    def acquire(self, experiment: str, target: str, **parameters: Any) -> dict:
        if target not in self.qubits:
            raise KeyError(f"Unknown qubit {target!r}")
        method = getattr(self, f"_acquire_{experiment}", None)
        if method is None:
            raise ValueError(f"Unsupported experiment: {experiment}")
        return method(self.qubits[target], **parameters)

    @staticmethod
    def _axis(parameters: dict, name: str) -> np.ndarray:
        values = np.asarray(parameters[name], dtype=float)
        if values.ndim != 1 or values.size < 3:
            raise ValueError(f"{name} must be a one-dimensional sweep with >= 3 points")
        if values.size > 20_000:
            raise ValueError(f"{name} sweep is too large")
        return values

    def _complex_noise(self, shape: tuple[int, ...], scale: float = 0.004) -> np.ndarray:
        sigma = scale * self.noise_scale
        return self.rng.normal(0, sigma, shape) + 1j * self.rng.normal(0, sigma, shape)

    @staticmethod
    def _notch(freq: np.ndarray, center: float, linewidth: float, depth: float = 0.65):
        x = 2 * (freq - center) / linewidth
        cable_phase = np.exp(1j * (0.15 + 2e-9 * (freq - center)))
        return cable_phase * (1 - depth / (1 + 1j * x))

    def _acquire_resonator_spectroscopy(self, truth: QubitTruth, **parameters: Any) -> dict:
        freq = self._axis(parameters, "frequencies_hz")
        center = truth.resonator_frequency_hz + self.rng.normal(0, 0.03e6)
        signal = self._notch(freq, center, truth.resonator_linewidth_hz)
        return {"frequencies_hz": freq, "s21": signal + self._complex_noise(freq.shape)}

    def _acquire_resonator_punchout(self, truth: QubitTruth, **parameters: Any) -> dict:
        freq = self._axis(parameters, "frequencies_hz")
        powers = self._axis(parameters, "powers_dbm")
        if np.max(powers) > 5:
            raise ValueError("Unsafe simulated readout power: maximum is +5 dBm")
        rows = []
        for power in powers:
            saturation = 1 / (1 + np.exp(-(power + 11) / 3.0))
            center = (
                truth.resonator_frequency_hz * (1 - saturation)
                + truth.bare_resonator_frequency_hz * saturation
            )
            depth = 0.75 - 0.25 * saturation
            rows.append(self._notch(freq, center, truth.resonator_linewidth_hz, depth))
        s21 = np.asarray(rows)
        return {
            "frequencies_hz": freq,
            "powers_dbm": powers,
            "s21": s21 + self._complex_noise(s21.shape, 0.006),
        }

    def _acquire_resonator_flux(self, truth: QubitTruth, **parameters: Any) -> dict:
        freq = self._axis(parameters, "frequencies_hz")
        bias = self._axis(parameters, "flux_bias")
        if np.min(bias) < -0.5 or np.max(bias) > 0.5:
            raise ValueError("Flux bias outside the configured safe range [-0.5, 0.5]")
        rows = []
        resonance = []
        for value in bias:
            phase = 2 * np.pi * (value - truth.flux_sweetspot) / truth.flux_period
            qubit_shift = truth.flux_dispersion_hz * (1 - np.cos(phase)) / 2
            dispersive_shift = 6e6 / (1 + qubit_shift / 80e6)
            center = truth.bare_resonator_frequency_hz - dispersive_shift
            resonance.append(center)
            rows.append(self._notch(freq, center, truth.resonator_linewidth_hz, 0.7))
        s21 = np.asarray(rows)
        return {
            "frequencies_hz": freq,
            "flux_bias": bias,
            "s21": s21 + self._complex_noise(s21.shape, 0.006),
            "latent_resonance_hz": np.asarray(resonance),
        }

    def _acquire_qubit_spectroscopy(self, truth: QubitTruth, **parameters: Any) -> dict:
        freq = self._axis(parameters, "frequencies_hz")
        halfwidth = truth.qubit_linewidth_hz / 2
        peak = halfwidth**2 / ((freq - truth.qubit_frequency_hz) ** 2 + halfwidth**2)
        population = 0.04 + 0.82 * peak + self.rng.normal(0, 0.018 * self.noise_scale, freq.size)
        return {"frequencies_hz": freq, "population": np.clip(population, 0, 1)}

    def _acquire_iq_raw(self, truth: QubitTruth, **parameters: Any) -> dict:
        shots = int(parameters.get("shots", 5000))
        if not 100 <= shots <= 1_000_000:
            raise ValueError("shots must be between 100 and 1,000,000")
        angle = 0.58
        sigma = 0.22 * self.noise_scale
        separation = 1.15 + 2.5 * (truth.readout_fidelity - 0.9)
        direction = np.array([np.cos(angle), np.sin(angle)])
        center0 = -0.5 * separation * direction
        center1 = 0.5 * separation * direction
        iq0 = self.rng.normal(center0, sigma, size=(shots, 2))
        iq1 = self.rng.normal(center1, sigma, size=(shots, 2))
        thermal = self.rng.random(shots) < truth.thermal_population
        iq0[thermal] += separation * direction
        decay = self.rng.random(shots) < 0.035
        fraction = self.rng.uniform(0.1, 0.9, decay.sum())[:, None]
        iq1[decay] -= fraction * separation * direction
        leakage = self.rng.random(shots) < 0.004
        iq1[leakage] += np.array([-0.25, 0.42])
        return {"iq_state0": iq0, "iq_state1": iq1, "shots": shots}

    def _acquire_rabi(self, truth: QubitTruth, **parameters: Any) -> dict:
        amp = self._axis(parameters, "amplitudes")
        envelope = np.exp(-np.abs(amp) / 4.0)
        population = 0.5 - 0.47 * np.cos(np.pi * amp / truth.pi_amplitude) * envelope
        population += self.rng.normal(0, 0.014 * self.noise_scale, amp.size)
        return {"amplitudes": amp, "population": np.clip(population, 0, 1)}

    def _acquire_ramsey(self, truth: QubitTruth, **parameters: Any) -> dict:
        delay = self._axis(parameters, "delays_s")
        drive = float(parameters["drive_frequency_hz"])
        detuning = drive - truth.qubit_frequency_hz
        population = 0.5 + 0.44 * np.cos(2 * np.pi * detuning * delay + 0.2) * np.exp(
            -delay / truth.t2_star_s
        )
        population += self.rng.normal(0, 0.012 * self.noise_scale, delay.size)
        return {"delays_s": delay, "population": np.clip(population, 0, 1)}

    def _acquire_t1(self, truth: QubitTruth, **parameters: Any) -> dict:
        delay = self._axis(parameters, "delays_s")
        population = 0.04 + 0.91 * np.exp(-delay / truth.t1_s)
        population += self.rng.normal(0, 0.011 * self.noise_scale, delay.size)
        return {"delays_s": delay, "population": np.clip(population, 0, 1)}

    def _acquire_drag(self, truth: QubitTruth, **parameters: Any) -> dict:
        beta = self._axis(parameters, "betas")
        repetitions = int(parameters.get("repetitions", 20))
        # A repeated-pulse error-amplification signal, normalized to stay in range.
        error_signal = 0.10 + (0.22 + repetitions / 100) * (beta - truth.drag_beta) ** 2
        error_signal += self.rng.normal(0, 0.009 * self.noise_scale, beta.size)
        return {"betas": beta, "error_signal": np.clip(error_signal, 0, 1)}

    def _acquire_single_qubit_xeb(self, truth: QubitTruth, **parameters: Any) -> dict:
        depths = self._axis(parameters, "depths").astype(int)
        circuits = int(parameters.get("circuits", 80))
        shots = int(parameters.get("shots", 1000))
        if circuits < 10 or shots < 100:
            raise ValueError("XEB needs at least 10 circuits and 100 shots")
        ideal_p1 = np.empty((depths.size, circuits))
        measured_p1 = np.empty_like(ideal_p1)
        for row, depth in enumerate(depths):
            # Haar-random single-qubit output probabilities are uniform on [0, 1].
            ideal = self.rng.uniform(0, 1, circuits)
            circuit_fidelity = truth.single_qubit_gate_fidelity ** int(depth)
            noisy = 0.5 + circuit_fidelity * (ideal - 0.5)
            counts = self.rng.binomial(shots, np.clip(noisy, 0, 1))
            ideal_p1[row] = ideal
            measured_p1[row] = counts / shots
        return {
            "depths": depths,
            "ideal_p1": ideal_p1,
            "measured_p1": measured_p1,
            "circuits": circuits,
            "shots": shots,
        }

    def _acquire_single_qubit_rb(self, truth: QubitTruth, **parameters: Any) -> dict:
        lengths = self._axis(parameters, "lengths").astype(int)
        sequences = int(parameters.get("sequences", 40))
        shots = int(parameters.get("shots", 1000))
        decay = 2 * truth.single_qubit_gate_fidelity - 1
        ideal_survival = 0.5 + 0.48 * decay**lengths
        survival = np.empty((lengths.size, sequences))
        for row, probability in enumerate(ideal_survival):
            survival[row] = self.rng.binomial(shots, probability, sequences) / shots
        return {"lengths": lengths, "survival": survival, "sequences": sequences, "shots": shots}
