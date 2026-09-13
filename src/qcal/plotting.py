"""Publication-friendly plots for raw measurements and fitted results."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _save(fig: plt.Figure, path: str | Path) -> str:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return str(output)


def plot_result(experiment: str, data: dict, fit: dict, path: str | Path) -> str:
    if experiment == "resonator_spectroscopy":
        return _plot_s21(data, fit, path)
    if experiment in {"resonator_punchout", "resonator_flux"}:
        return _plot_2d(experiment, data, fit, path)
    if experiment == "iq_raw":
        return _plot_iq(data, fit, path)
    if experiment in {"qubit_spectroscopy", "rabi", "ramsey", "t1", "drag"}:
        return _plot_curve(experiment, data, fit, path)
    if experiment in {"single_qubit_xeb", "single_qubit_rb"}:
        return _plot_benchmark(experiment, data, fit, path)
    raise ValueError(f"No plotter for {experiment}")


def _plot_s21(data: dict, fit: dict, path: str | Path) -> str:
    frequency = np.asarray(data["frequencies_hz"]) / 1e9
    signal = np.asarray(data["s21"])
    fig, axes = plt.subplots(2, 1, figsize=(7.5, 6), sharex=True)
    axes[0].plot(frequency, 20 * np.log10(np.abs(signal)), color="#136f9a")
    axes[0].axvline(fit["resonator_frequency_hz"] / 1e9, color="#d1495b", ls="--", label="fitted fᵣ")
    axes[0].set_ylabel("|S21| (dB)"); axes[0].legend()
    axes[1].plot(frequency, np.unwrap(np.angle(signal)), color="#2a9d8f")
    axes[1].set(xlabel="Readout frequency (GHz)", ylabel="Phase (rad)")
    fig.suptitle("Resonator spectroscopy")
    return _save(fig, path)


def _plot_2d(experiment: str, data: dict, fit: dict, path: str | Path) -> str:
    frequency = np.asarray(data["frequencies_hz"]) / 1e9
    if experiment == "resonator_punchout":
        y = np.asarray(data["powers_dbm"]); ylabel = "Readout power (dBm)"
        trace = np.asarray(fit["resonance_by_power_hz"]) / 1e9
        selected = fit["selected_power_dbm"]; title = "Resonator punchout / S21 power2d"
    else:
        y = np.asarray(data["flux_bias"]); ylabel = "Flux bias (arb.)"
        trace = np.asarray(fit["resonance_by_flux_hz"]) / 1e9
        selected = fit["flux_sweetspot"]; title = "Resonator flux map / ZPA2D"
    magnitude = 20 * np.log10(np.abs(data["s21"]))
    fig, ax = plt.subplots(figsize=(8, 5.5))
    mesh = ax.pcolormesh(frequency, y, magnitude, shading="auto", cmap="viridis")
    ax.plot(trace, y, color="white", lw=1.4, label="extracted resonance")
    ax.axhline(selected, color="#ffb703", ls="--", label="selected operating point")
    ax.set(xlabel="Readout frequency (GHz)", ylabel=ylabel, title=title)
    fig.colorbar(mesh, ax=ax, label="|S21| (dB)"); ax.legend(loc="best")
    return _save(fig, path)


def _plot_iq(data: dict, fit: dict, path: str | Path) -> str:
    iq0, iq1 = np.asarray(data["iq_state0"]), np.asarray(data["iq_state1"])
    p0, p1 = np.asarray(fit["projected_state0"]), np.asarray(fit["projected_state1"])
    angle, threshold = fit["rotation_angle_rad"], fit["threshold"]
    direction = np.array([np.cos(angle), np.sin(angle)])
    midpoint = .5 * (np.asarray(fit["center_state0"]) + np.asarray(fit["center_state1"]))
    perpendicular = np.array([-direction[1], direction[0]])
    boundary_center = midpoint + (threshold - midpoint @ direction) * direction
    span = 2.2 * max(np.std(iq0), np.std(iq1))
    line = boundary_center + np.linspace(-span, span, 100)[:, None] * perpendicular
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes[0, 0].scatter(iq0[:, 0], iq0[:, 1], s=3, alpha=.22, label="prepared |0⟩")
    axes[0, 0].scatter(iq1[:, 0], iq1[:, 1], s=3, alpha=.22, label="prepared |1⟩")
    axes[0, 0].plot(line[:, 0], line[:, 1], "k--", lw=1.4, label="decision boundary")
    axes[0, 0].set(xlabel="I", ylabel="Q", title="Raw single-shot IQ"); axes[0, 0].legend(markerscale=3)
    axes[0, 1].hist2d(np.r_[iq0[:, 0], iq1[:, 0]], np.r_[iq0[:, 1], iq1[:, 1]], bins=80, cmap="magma")
    axes[0, 1].plot(line[:, 0], line[:, 1], "w--", lw=1.3)
    axes[0, 1].set(xlabel="I", ylabel="Q", title="IQ density and boundary")
    axes[1, 0].hist(p0, bins=80, alpha=.65, density=True, label="|0⟩")
    axes[1, 0].hist(p1, bins=80, alpha=.65, density=True, label="|1⟩")
    axes[1, 0].axvline(threshold, color="black", ls="--", label="threshold")
    axes[1, 0].set(xlabel="Rotated I", ylabel="Density", title="1D assignment histogram"); axes[1, 0].legend()
    confusion = np.asarray(fit["confusion_matrix"])
    image = axes[1, 1].imshow(confusion / confusion.sum(axis=1, keepdims=True), vmin=0, vmax=1, cmap="Blues")
    for row in range(2):
        for column in range(2):
            axes[1, 1].text(column, row, f"{confusion[row, column]}\n({confusion[row, column]/confusion[row].sum():.1%})", ha="center", va="center")
    axes[1, 1].set(xticks=[0, 1], yticks=[0, 1], xlabel="Assigned", ylabel="Prepared", title=f"Confusion matrix, F={fit['assignment_fidelity']:.3f}")
    fig.colorbar(image, ax=axes[1, 1], fraction=.046)
    fig.suptitle("IQ raw readout calibration")
    return _save(fig, path)


def _plot_curve(experiment: str, data: dict, fit: dict, path: str | Path) -> str:
    mapping = {
        "qubit_spectroscopy": ("frequencies_hz", "population", "fitted_population", 1e9, "Drive frequency (GHz)", "Qubit spectroscopy"),
        "rabi": ("amplitudes", "population", "fitted_population", 1, "Drive amplitude", "Rabi amplitude calibration"),
        "ramsey": ("delays_s", "population", "fitted_population", 1e-6, "Delay (µs)", "Ramsey / frequency and T2*"),
        "t1": ("delays_s", "population", "fitted_population", 1e-6, "Delay (µs)", "Energy relaxation T1"),
        "drag": ("betas", "error_signal", "fitted_signal", 1, "DRAG beta", "DRAG calibration"),
    }
    xkey, ykey, fitted_key, scale, xlabel, title = mapping[experiment]
    x, y = np.asarray(data[xkey]) / scale, np.asarray(data[ykey])
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.scatter(x, y, s=18, alpha=.7, label="measurement")
    ax.plot(x, fit[fitted_key], color="#d1495b", lw=2, label="fit")
    ax.set(xlabel=xlabel, ylabel="Population / signal", title=title); ax.legend(); ax.grid(alpha=.2)
    return _save(fig, path)


def _plot_benchmark(experiment: str, data: dict, fit: dict, path: str | Path) -> str:
    fig, ax = plt.subplots(figsize=(7.5, 5))
    if experiment == "single_qubit_xeb":
        x = np.asarray(data["depths"]); y = np.asarray(fit["xeb_fidelity_by_depth"]); predicted = fit["fitted_decay"]
        title = f"Single-qubit XEB, per-gate fidelity={fit['per_gate_fidelity']:.5f}"; ylabel = "XEB fidelity"
    else:
        x = np.asarray(data["lengths"]); y = np.asarray(fit["mean_survival"]); predicted = fit["fitted_survival"]
        title = f"Single-qubit RB, avg. fidelity={fit['average_gate_fidelity']:.5f}"; ylabel = "Survival probability"
    ax.scatter(x, y, label="measurement"); ax.plot(x, predicted, color="#d1495b", label="decay fit")
    ax.set(xlabel="Circuit depth", ylabel=ylabel, title=title); ax.grid(alpha=.2); ax.legend()
    return _save(fig, path)
