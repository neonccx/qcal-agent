# Real-hardware integration boundary

Version 1.0 implements single-qubit calibration only. It intentionally does not
provide a coupler object, `generate_coupler`, two-qubit gate calibration, or any
vendor-private API. Power2D optimizes the resonator/readout operating power;
ZPA2D maps the selected qubit's flux-bias coordinate against S21. Neither implies
that a tunable coupler exists.

The model-facing names and measurement-suite equivalents are:

| Agent action | Measurement suite | Intended lab operation |
|---|---|---|
| `sq.s21` | `resonator_spectroscopy` | one-dimensional resonator S21 |
| `sq.s21_power2d` | `resonator_punchout` | frequency-power S21 map |
| `sq.s21_zpa2d` | `resonator_flux` | frequency-flux S21 map |
| `sq.spectroscopy` | `qubit_spectroscopy` | qubit drive spectroscopy |
| `sq.piamp` | `rabi` | pi-amplitude calibration |
| `sq.ramsey_df` | `ramsey` | detuning and T2-star |
| `sq.t1` | `t1` | energy relaxation |
| `sq.xeb` | `single_qubit_xeb` | single-qubit randomized-circuit proxy |
| `sq.iqraw` | `iq_raw` | raw state-0/state-1 single shots |

To connect a real stack, instantiate `qmagent.hardware_backend.RegisteredHardwareBackend`
with an explicit allow-list. Each handler receives `{tool, state, scan, sequence}`
and returns raw I/Q data. The adapter performs deterministic analysis afterward;
the language model never receives instrument objects or arbitrary call access.

Before a supervised hardware run, replace simulation bounds with laboratory limits,
validate units and channel mappings, require an exclusive device lock, implement
timeouts and safe shutdown, test readback after every write, and keep action-time
human approval enabled. A simulator pass is not evidence that these controls work
on a real dilution refrigerator.

The independent `qcal full` workflow stops at the first unreliable fit. It writes
`workflow_status.json` and `calibration_checkpoint.json`; only a complete,
all-success run writes `final_calibration.json`. Its CLI returns a nonzero exit
code on a warning and requires an empty output directory so a stale final result
cannot be mistaken for a new one. Treat these artifacts as simulation evidence, not hardware
qualification. The registered hardware adapter's acquisition sequence is
monotonic even after a logical restore, because physical acquisitions cannot be
undone.

`qcal` IQraw trains the threshold on 70% of each prepared state's shots and
reports confusion and assignment fidelity on the remaining 30%. Its
`apparent_state0_excited_fraction` includes preparation and readout errors; the
routine intentionally does not report qubit temperature. Its single-qubit XEB
backend currently samples output probabilities rather than executing explicit
random gate sequences, so the fitted decay is a simulator proxy. Google Cirq's
[XEB definition](https://quantumai.google/reference/python/cirq/xeb_fidelity)
instead begins with a specified circuit and measured bitstrings.

The `qcal full` order puts Rabi before IQraw so the intended |1> preparation
can use a calibrated pi pulse in a future hardware implementation. The current
simulator does not yet model the resulting preparation error as a function of
the committed pi amplitude; this ordering alone is not hardware validation.
