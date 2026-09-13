"""Device and calibration state models.

The simulator keeps unknown physical truth separate from the values known by the
calibration workflow. A real backend would only expose ``QubitCalibration``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass(slots=True)
class QubitTruth:
    qubit_id: str = "Q0"
    resonator_frequency_hz: float = 6.72e9
    bare_resonator_frequency_hz: float = 6.726e9
    resonator_linewidth_hz: float = 2.2e6
    qubit_frequency_hz: float = 4.85e9
    qubit_linewidth_hz: float = 2.5e6
    flux_sweetspot: float = 0.032
    flux_period: float = 1.0
    flux_dispersion_hz: float = 260e6
    readout_power_dbm: float = -18.0
    pi_amplitude: float = 0.42
    drag_beta: float = -0.16
    t1_s: float = 42e-6
    t2_star_s: float = 25e-6
    readout_fidelity: float = 0.965
    single_qubit_gate_fidelity: float = 0.993
    thermal_population: float = 0.025


@dataclass(slots=True)
class QubitCalibration:
    qubit_id: str = "Q0"
    resonator_frequency_hz: float = 6.70e9
    qubit_frequency_hz: float = 4.80e9
    readout_power_dbm: float = -25.0
    flux_bias: float = 0.0
    pi_amplitude: float = 0.35
    drag_beta: float = 0.0
    iq_rotation_rad: float = 0.0
    iq_threshold: float = 0.0
    t1_s: float | None = None
    t2_star_s: float | None = None
    readout_fidelity: float | None = None
    gate_fidelity: float | None = None
    history: list[dict] = field(default_factory=list)

    def update(self, source: str, **changes: float) -> None:
        accepted = {}
        for name, value in changes.items():
            if not hasattr(self, name):
                raise AttributeError(f"Unknown calibration parameter: {name}")
            setattr(self, name, value)
            accepted[name] = value
        self.history.append({"source": source, "changes": accepted})

    def to_dict(self) -> dict:
        return asdict(self)
