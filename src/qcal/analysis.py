"""Deterministic analysis and fitting routines for calibration experiments."""

from __future__ import annotations

import numpy as np
from scipy.optimize import curve_fit


def _r2(observed: np.ndarray, predicted: np.ndarray) -> float:
    residual = float(np.sum((observed - predicted) ** 2))
    total = float(np.sum((observed - np.mean(observed)) ** 2))
    return 1.0 - residual / total if total > 0 else 0.0


def analyze_resonator_spectroscopy(data: dict) -> tuple[dict, dict, dict]:
    frequency = np.asarray(data["frequencies_hz"])
    magnitude = np.abs(data["s21"])
    index = int(np.argmin(magnitude))
    center = float(frequency[index])
    baseline = float(np.percentile(magnitude, 90))
    minimum = float(magnitude[index])
    half = minimum + (baseline - minimum) / 2
    below = np.flatnonzero(magnitude <= half)
    linewidth = float(frequency[below[-1]] - frequency[below[0]]) if below.size > 1 else 0.0
    contrast = (baseline - minimum) / max(baseline, 1e-12)
    fit = {"resonator_frequency_hz": center, "linewidth_hz": linewidth, "baseline": baseline}
    quality = {"contrast": float(contrast), "fit_ok": bool(contrast > 0.08)}
    updates = {"resonator_frequency_hz": center}
    return fit, quality, updates


def analyze_resonator_punchout(data: dict) -> tuple[dict, dict, dict]:
    frequency = np.asarray(data["frequencies_hz"])
    powers = np.asarray(data["powers_dbm"])
    magnitude = np.abs(data["s21"])
    indices = np.argmin(magnitude, axis=1)
    centers = frequency[indices]
    contrast = np.percentile(magnitude, 90, axis=1) - np.min(magnitude, axis=1)
    low_power_center = float(np.median(centers[: max(2, len(centers) // 4)]))
    shift = np.abs(centers - low_power_center)
    safe = shift < max(2.5e6, 1.5 * np.median(np.abs(np.diff(centers))))
    score = contrast / np.maximum(np.median(contrast), 1e-12)
    candidates = np.flatnonzero(safe & (score > 0.75))
    chosen = int(candidates[-1]) if candidates.size else int(np.argmax(contrast))
    fit = {
        "resonance_by_power_hz": centers,
        "contrast_by_power": contrast,
        "selected_power_dbm": float(powers[chosen]),
        "selected_resonator_frequency_hz": float(centers[chosen]),
        "nonlinear_shift_hz": shift,
    }
    quality = {"max_contrast": float(np.max(contrast)), "fit_ok": bool(np.max(contrast) > 0.05)}
    updates = {
        "readout_power_dbm": float(powers[chosen]),
        "resonator_frequency_hz": float(centers[chosen]),
    }
    return fit, quality, updates


def analyze_resonator_flux(data: dict) -> tuple[dict, dict, dict]:
    frequency = np.asarray(data["frequencies_hz"])
    bias = np.asarray(data["flux_bias"])
    magnitude = np.abs(data["s21"])
    centers = frequency[np.argmin(magnitude, axis=1)]
    rough = int(np.argmin(centers))
    radius = min(5, rough, len(bias) - rough - 1)
    if radius >= 2:
        subset = slice(rough - radius, rough + radius + 1)
        coeff = np.polyfit(bias[subset], centers[subset], 2)
        sweetspot = float(-coeff[1] / (2 * coeff[0])) if coeff[0] > 0 else float(bias[rough])
        sweetspot = float(np.clip(sweetspot, bias[rough - radius], bias[rough + radius]))
        fitted = np.polyval(coeff, bias)
        r_squared = _r2(centers[subset], np.polyval(coeff, bias[subset]))
    else:
        coeff = np.array([0.0, 0.0, centers[rough]])
        sweetspot = float(bias[rough])
        fitted = np.full_like(bias, centers[rough])
        r_squared = 0.0
    resonator = float(np.interp(sweetspot, bias, centers))
    fit = {
        "resonance_by_flux_hz": centers,
        "quadratic_curve_hz": fitted,
        "quadratic_coefficients": coeff,
        "flux_sweetspot": sweetspot,
        "resonator_frequency_hz": resonator,
    }
    quality = {"r_squared": float(r_squared), "fit_ok": bool(abs(sweetspot) <= 0.5)}
    return fit, quality, {"flux_bias": sweetspot, "resonator_frequency_hz": resonator}


def _lorentzian(x: np.ndarray, offset: float, amplitude: float, center: float, width: float):
    return offset + amplitude * width**2 / ((x - center) ** 2 + width**2)


def analyze_qubit_spectroscopy(data: dict) -> tuple[dict, dict, dict]:
    frequency = np.asarray(data["frequencies_hz"], dtype=float)
    population = np.asarray(data["population"], dtype=float)
    p0 = [float(np.min(population)), float(np.ptp(population)), float(frequency[np.argmax(population)]), 2e6]
    bounds = ([0, 0, frequency.min(), 1e3], [1, 1.5, frequency.max(), np.ptp(frequency)])
    params, _ = curve_fit(_lorentzian, frequency, population, p0=p0, bounds=bounds, maxfev=20000)
    predicted = _lorentzian(frequency, *params)
    fit = {
        "offset": float(params[0]), "amplitude": float(params[1]),
        "qubit_frequency_hz": float(params[2]), "linewidth_hz": float(2 * params[3]),
        "fitted_population": predicted,
    }
    quality = {"r_squared": _r2(population, predicted), "fit_ok": bool(_r2(population, predicted) > 0.75)}
    return fit, quality, {"qubit_frequency_hz": float(params[2])}


def analyze_iq_raw(data: dict, qubit_frequency_hz: float | None = None) -> tuple[dict, dict, dict]:
    """Train an IQ discriminator and evaluate it on shots not used to fit it.

    ``qubit_frequency_hz`` is retained for API compatibility. A nominal-|0>
    assignment error is not an identifiable thermal population or temperature.
    """
    iq0 = np.asarray(data["iq_state0"], dtype=float)
    iq1 = np.asarray(data["iq_state1"], dtype=float)
    if (iq0.ndim != 2 or iq1.ndim != 2 or iq0.shape[1] != 2 or iq1.shape[1] != 2
            or min(len(iq0), len(iq1)) < 100 or not np.isfinite(iq0).all()
            or not np.isfinite(iq1).all()):
        raise ValueError("IQraw requires at least 100 finite I/Q shots per prepared state")
    train0, train1 = int(0.7 * len(iq0)), int(0.7 * len(iq1))
    center0, center1 = iq0[:train0].mean(axis=0), iq1[:train1].mean(axis=0)
    direction = center1 - center0
    separation = float(np.linalg.norm(direction))
    direction = direction / separation if separation > 1e-12 else np.array([1.0, 0.0])
    projected0, projected1 = iq0 @ direction, iq1 @ direction
    values = np.concatenate([projected0[:train0], projected1[:train1]])
    labels = np.concatenate([np.zeros(train0, dtype=int), np.ones(train1, dtype=int)])
    order = np.argsort(values)
    sorted_labels = labels[order]
    sorted_values = values[order]
    errors = np.cumsum(sorted_labels == 1)[:-1] / train1 + (
        train0 - np.cumsum(sorted_labels == 0)[:-1]
    ) / train0
    valid_cuts = sorted_values[:-1] < sorted_values[1:]
    if valid_cuts.any():
        cut = int(np.argmin(np.where(valid_cuts, errors, np.inf)))
        threshold = float((sorted_values[cut] + sorted_values[cut + 1]) / 2)
    else:
        threshold = float(sorted_values[0])
    heldout0, heldout1 = projected0[train0:], projected1[train1:]
    pred0, pred1 = heldout0 > threshold, heldout1 > threshold
    confusion = np.array([[np.sum(~pred0), np.sum(pred0)], [np.sum(~pred1), np.sum(pred1)]], dtype=int)
    p10, p01 = float(np.mean(pred0)), float(np.mean(~pred1))
    fidelity = 1 - 0.5 * (p10 + p01)
    pooled_sigma = np.sqrt(0.5 * (np.var(heldout0) + np.var(heldout1)))
    snr = abs(np.mean(heldout1) - np.mean(heldout0)) / max(pooled_sigma, 1e-12)
    fit = {
        "center_state0": center0, "center_state1": center1, "rotation_angle_rad": float(np.arctan2(direction[1], direction[0])),
        "threshold": threshold, "projected_state0": projected0, "projected_state1": projected1,
        "confusion_matrix": confusion, "assignment_fidelity": float(fidelity), "snr": float(snr),
        "apparent_state0_excited_fraction": p10,
        "training_shots_per_state": [train0, train1],
        "validation_shots_per_state": [len(heldout0), len(heldout1)],
    }
    quality = {"assignment_fidelity": float(fidelity), "snr": float(snr),
               "validation": "held_out_30_percent", "fit_ok": bool(fidelity > 0.85 and separation > 1e-12)}
    return fit, quality, {"iq_rotation_rad": fit["rotation_angle_rad"], "iq_threshold": threshold, "readout_fidelity": float(fidelity)}


def _rabi_model(x, offset, amplitude, pi_amplitude, decay, phase):
    return offset + amplitude * np.cos(np.pi * x / pi_amplitude + phase) * np.exp(-np.abs(x) / decay)


def analyze_rabi(data: dict) -> tuple[dict, dict, dict]:
    x, y = np.asarray(data["amplitudes"]), np.asarray(data["population"])
    guess_pi = float(x[np.argmax(y)])
    guess_pi = guess_pi if guess_pi > 0 else float(np.ptp(x) / 2)
    params, _ = curve_fit(_rabi_model, x, y, p0=[0.5, -0.47, guess_pi, 4.0, 0],
                          bounds=([0, -1, 0.02, 0.05, -np.pi], [1, 1, max(x.max() * 2, .1), 20, np.pi]), maxfev=30000)
    predicted = _rabi_model(x, *params)
    pi_amp = float(abs(params[2]))
    fit = {"offset": float(params[0]), "amplitude": float(params[1]), "pi_amplitude": pi_amp,
           "decay_scale": float(params[3]), "phase_rad": float(params[4]), "fitted_population": predicted}
    quality = {"r_squared": _r2(y, predicted), "fit_ok": bool(_r2(y, predicted) > 0.75)}
    return fit, quality, {"pi_amplitude": pi_amp}


def _ramsey_model(t, offset, amplitude, detuning, t2star, phase):
    return offset + amplitude * np.cos(2 * np.pi * detuning * t + phase) * np.exp(-t / t2star)


def analyze_ramsey(data: dict, drive_frequency_hz: float) -> tuple[dict, dict, dict]:
    t, y = np.asarray(data["delays_s"]), np.asarray(data["population"])
    spectrum = np.abs(np.fft.rfft((y - y.mean()) * np.hanning(len(y))))
    frequencies = np.fft.rfftfreq(len(t), float(np.mean(np.diff(t))))
    guess = float(frequencies[1 + np.argmax(spectrum[1:])])
    params, _ = curve_fit(_ramsey_model, t, y, p0=[.5, .4, guess, max(t.max() / 3, 1e-6), 0],
                          bounds=([0, -1, 0, 1e-7, -np.pi], [1, 1, frequencies[-1], max(10 * t.max(), 1e-5), np.pi]), maxfev=30000)
    predicted = _ramsey_model(t, *params)
    corrected = float(drive_frequency_hz - params[2])
    fit = {"offset": float(params[0]), "amplitude": float(params[1]), "detuning_hz": float(params[2]),
           "t2_star_s": float(params[3]), "phase_rad": float(params[4]), "qubit_frequency_hz": corrected,
           "fitted_population": predicted}
    quality = {"r_squared": _r2(y, predicted), "fit_ok": bool(_r2(y, predicted) > .65)}
    return fit, quality, {"qubit_frequency_hz": corrected, "t2_star_s": float(params[3])}


def _exponential(t, offset, amplitude, lifetime):
    return offset + amplitude * np.exp(-t / lifetime)


def analyze_t1(data: dict) -> tuple[dict, dict, dict]:
    t, y = np.asarray(data["delays_s"]), np.asarray(data["population"])
    params, _ = curve_fit(_exponential, t, y, p0=[y[-1], y[0] - y[-1], max(t.max() / 4, 1e-6)],
                          bounds=([0, 0, 1e-8], [1, 1.5, max(10 * t.max(), 1e-5)]), maxfev=20000)
    predicted = _exponential(t, *params)
    fit = {"offset": float(params[0]), "amplitude": float(params[1]), "t1_s": float(params[2]), "fitted_population": predicted}
    quality = {"r_squared": _r2(y, predicted), "fit_ok": bool(_r2(y, predicted) > .8)}
    return fit, quality, {"t1_s": float(params[2])}


def analyze_drag(data: dict) -> tuple[dict, dict, dict]:
    beta, signal = np.asarray(data["betas"]), np.asarray(data["error_signal"])
    coeff = np.polyfit(beta, signal, 2)
    optimum = float(-coeff[1] / (2 * coeff[0]))
    optimum = float(np.clip(optimum, beta.min(), beta.max()))
    predicted = np.polyval(coeff, beta)
    fit = {"drag_beta": optimum, "quadratic_coefficients": coeff, "fitted_signal": predicted}
    quality = {"r_squared": _r2(signal, predicted), "fit_ok": bool(coeff[0] > 0 and _r2(signal, predicted) > .6)}
    return fit, quality, {"drag_beta": optimum}


def analyze_xeb(data: dict) -> tuple[dict, dict, dict]:
    depths = np.asarray(data["depths"], dtype=float)
    ideal, measured = np.asarray(data["ideal_p1"]), np.asarray(data["measured_p1"])
    centered_ideal, centered_measured = ideal - .5, measured - .5
    fidelity = np.sum(centered_ideal * centered_measured, axis=1) / np.maximum(np.sum(centered_ideal**2, axis=1), 1e-12)
    positive = np.clip(fidelity, 1e-6, 1)
    slope, intercept = np.polyfit(depths, np.log(positive), 1)
    per_gate = float(np.clip(np.exp(slope), 0, 1))
    predicted = np.exp(intercept + slope * depths)
    fit = {"xeb_fidelity_by_depth": fidelity, "per_gate_fidelity": per_gate, "fitted_decay": predicted}
    quality = {"decay_r_squared": _r2(positive, predicted), "fit_ok": bool(per_gate > .8)}
    return fit, quality, {"gate_fidelity": per_gate}


def analyze_rb(data: dict) -> tuple[dict, dict, dict]:
    lengths = np.asarray(data["lengths"], dtype=float)
    survival = np.asarray(data["survival"]).mean(axis=1)
    centered = np.clip(survival - .5, 1e-6, 1)
    slope, intercept = np.polyfit(lengths, np.log(centered), 1)
    decay = float(np.clip(np.exp(slope), 0, 1))
    predicted = .5 + np.exp(intercept + slope * lengths)
    average_fidelity = float((1 + decay) / 2)
    fit = {"mean_survival": survival, "decay_parameter": decay, "average_gate_fidelity": average_fidelity, "fitted_survival": predicted}
    quality = {"decay_r_squared": _r2(survival, predicted), "fit_ok": bool(average_fidelity > .8)}
    return fit, quality, {"gate_fidelity": average_fidelity}
