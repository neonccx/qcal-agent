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
