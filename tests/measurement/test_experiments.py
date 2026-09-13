from pathlib import Path

import numpy as np
import pytest

from qcal import CalibrationWorkflow, ExperimentSuite, QubitCalibration, SimulatedBackend


@pytest.fixture
def suite(tmp_path: Path):
    backend = SimulatedBackend(seed=11)
    backend.connect()
    return ExperimentSuite(backend, QubitCalibration(), tmp_path)


def test_core_parameter_estimates(suite: ExperimentSuite):
    resonator = suite.resonator_spectroscopy()
    spectroscopy = suite.qubit_spectroscopy()
    rabi = suite.rabi()
    ramsey = suite.ramsey(drive_frequency_hz=4.8512e9)
    lifetime = suite.t1()
    assert resonator.fit["resonator_frequency_hz"] == pytest.approx(6.72e9, abs=0.4e6)
    assert spectroscopy.fit["qubit_frequency_hz"] == pytest.approx(4.85e9, abs=0.5e6)
    assert rabi.fit["pi_amplitude"] == pytest.approx(.42, abs=.015)
    assert ramsey.fit["t2_star_s"] == pytest.approx(25e-6, rel=.12)
    assert lifetime.fit["t1_s"] == pytest.approx(42e-6, rel=.08)


def test_iq_and_benchmarks(suite: ExperimentSuite):
    iq = suite.iq_raw(shots=3000)
    xeb = suite.single_qubit_xeb(circuits=120)
    rb = suite.single_qubit_rb(sequences=60)
    assert iq.fit["assignment_fidelity"] > .94
    assert np.asarray(iq.fit["confusion_matrix"]).shape == (2, 2)
    assert xeb.fit["per_gate_fidelity"] == pytest.approx(.993, abs=.003)
    assert rb.fit["average_gate_fidelity"] == pytest.approx(.993, abs=.003)


def test_full_workflow_persists_all_artifacts(tmp_path: Path):
    suite = ExperimentSuite(SimulatedBackend(seed=17), QubitCalibration(), tmp_path)
    results = CalibrationWorkflow(suite).run()
    assert len(results) == 11
    assert all((tmp_path / f"{result.experiment}.json").exists() for result in results)
    assert all((tmp_path / f"{result.experiment}.png").exists() for result in results)
    assert (tmp_path / "final_calibration.json").exists()
    assert suite.calibration.flux_bias == pytest.approx(.032, abs=.01)
    assert suite.calibration.drag_beta == pytest.approx(-.16, abs=.03)


def test_backend_safety_limits():
    backend = SimulatedBackend()
    with pytest.raises(ValueError, match="safe range"):
        backend.acquire("resonator_flux", "Q0", frequencies_hz=np.linspace(6.7e9, 6.74e9, 11), flux_bias=np.linspace(-.6, .2, 11))
