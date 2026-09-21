from pathlib import Path
import json

import numpy as np
import pytest

from qcal import CalibrationWorkflow, ExperimentSuite, QubitCalibration, SimulatedBackend
from qcal import analysis
from qcal.cli import main as qcal_main
from qcal.experiments import EXPERIMENTS


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
    assert np.asarray(iq.fit["confusion_matrix"]).sum() == 1800
    assert iq.fit["validation_shots_per_state"] == [900, 900]
    assert "estimated_temperature_k" not in iq.fit
    assert "estimated_thermal_population" not in iq.fit
    assert iq.quality["validation"] == "held_out_30_percent"
    assert xeb.fit["per_gate_fidelity"] == pytest.approx(.993, abs=.003)
    assert rb.fit["average_gate_fidelity"] == pytest.approx(.993, abs=.003)


def test_iq_discriminator_rejects_overlapping_clusters():
    from qcal.analysis import analyze_iq_raw

    shots = np.zeros((100, 2))
    _, quality, _ = analyze_iq_raw({"iq_state0": shots, "iq_state1": shots})
    assert quality["fit_ok"] is False


def test_full_workflow_persists_all_artifacts(tmp_path: Path):
    suite = ExperimentSuite(SimulatedBackend(seed=17), QubitCalibration(), tmp_path)
    results = CalibrationWorkflow(suite).run()
    assert len(results) == 11
    assert EXPERIMENTS.index("rabi") < EXPERIMENTS.index("iq_raw")
    assert all((tmp_path / f"{result.experiment}.json").exists() for result in results)
    assert all((tmp_path / f"{result.experiment}.png").exists() for result in results)
    assert (tmp_path / "final_calibration.json").exists()
    assert json.loads((tmp_path / "workflow_status.json").read_text())["status"] == "complete"
    assert suite.calibration.flux_bias == pytest.approx(.032, abs=.01)
    assert suite.calibration.drag_beta == pytest.approx(-.16, abs=.03)


def test_full_workflow_stops_on_unreliable_fit_and_exits_nonzero(tmp_path: Path, monkeypatch):
    original = analysis.analyze_resonator_spectroscopy

    def unreliable(data):
        fit, quality, updates = original(data)
        return fit, {**quality, "fit_ok": False}, updates

    monkeypatch.setattr(analysis, "analyze_resonator_spectroscopy", unreliable)
    assert qcal_main(["full", "--output", str(tmp_path), "--seed", "17"]) == 1
    assert (tmp_path / "resonator_spectroscopy.json").exists()
    assert not (tmp_path / "resonator_punchout.json").exists()
    assert not (tmp_path / "final_calibration.json").exists()
    assert (tmp_path / "calibration_checkpoint.json").exists()
    status = json.loads((tmp_path / "workflow_status.json").read_text())
    assert status["status"] == "blocked"
    assert status["stopped_at"] == "resonator_spectroscopy"
    assert status["completed_experiments"] == 1
    assert json.loads((tmp_path / "calibration_checkpoint.json").read_text())["history"] == []


def test_full_workflow_does_not_overwrite_previous_run(tmp_path: Path):
    keep = tmp_path / "previous-result.json"
    keep.write_text("keep", encoding="utf-8")
    suite = ExperimentSuite(SimulatedBackend(seed=17), QubitCalibration(), tmp_path)
    with pytest.raises(FileExistsError, match="must be empty"):
        CalibrationWorkflow(suite).run()
    assert keep.read_text(encoding="utf-8") == "keep"


def test_missing_fit_quality_cannot_silently_pass(tmp_path: Path, monkeypatch):
    original = analysis.analyze_resonator_spectroscopy

    def missing_quality(data):
        fit, quality, updates = original(data)
        quality.pop("fit_ok")
        return fit, quality, updates

    monkeypatch.setattr(analysis, "analyze_resonator_spectroscopy", missing_quality)
    suite = ExperimentSuite(SimulatedBackend(seed=17), QubitCalibration(), tmp_path)
    results = CalibrationWorkflow(suite).run()
    assert len(results) == 1
    assert results[0].status == "warning"
    assert not (tmp_path / "final_calibration.json").exists()


def test_backend_safety_limits():
    backend = SimulatedBackend()
    with pytest.raises(ValueError, match="safe range"):
        backend.acquire("resonator_flux", "Q0", frequencies_hz=np.linspace(6.7e9, 6.74e9, 11), flux_bias=np.linspace(-.6, .2, 11))


def test_flux_raw_observation_does_not_expose_simulator_truth(suite: ExperimentSuite):
    result = suite.resonator_flux()
    assert "latent_resonance_hz" not in result.raw_data
    assert "_truth" not in result.raw_data
