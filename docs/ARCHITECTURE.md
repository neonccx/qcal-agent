# Architecture and release boundary

The release has four explicit layers:

1. `qcal` acquires raw arrays and provides deterministic analyses and plots.
2. `qmagent` exposes only registered calibration actions to a policy model.
3. The controller validates units, ranges, prerequisites, budgets, fit reliability and rollback before state commits.
4. `training` learns the next registered action from public state and fit summaries; raw arrays and simulator truth are excluded from prompts.

The simulation backend and a future hardware backend must satisfy the same typed observation contract. A hardware adapter is not considered validated merely because it imports successfully: it requires device-specific limits, readback, timeout, lock, emergency-stop and supervised dry-run evidence.

Release acceptance requires unit tests, dataset audit, base-model frozen-context baseline, LoRA training, test/OOD comparison, fresh-seed closed-loop runs, artifact checksums and a downloadable GitHub release asset. Simulation evidence must be labelled as such.
