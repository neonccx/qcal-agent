# Candidate selection — 2026-09-15

The user cancelled the safety-only continuation on 2026-09-15. Do not restart
its training/evaluation or use its adapter in a release. Preserve its artifacts
for audit; cancellation does not remove runtime safety validation.

Selected candidate (not release-approved):

- Server run: `runs/qcal_1_0_repair_20260914_2302`
- Adapter: `training/final_adapter/adapter_model.safetensors`
- SHA256: `7aaaf32cac4bee2d4092731be0e986d54c77c690c586381570d2c9d45653b96e`
- Frozen test next-tool accuracy: 158/160 (98.75%).
- Frozen OOD controller executable rate: 152/160 (95%); release gate still fails.
- Synthetic closed-loop acceptance: 3/3; not hardware validation.

Rejected continuation: `runs/qcal_1_0_safety_20260915_1201`, adapter SHA256
`327c308464f087efe01d3056a3176e6ab847673dce47d8e7edcde3a831410147`.
Its frozen test next-tool accuracy fell to 111/160 (69.375%). The continuation
used 336 budget-zero and 219 Ramsey-repeat training examples without normal
workflow replay. These observations support action-bias/forgetting as the likely
explanation, not a controlled causal proof. OOD evaluation was interrupted.

Next implementation should address deterministic budget and calibration checks
in the controller while retaining the selected adapter. It requires separate
verification; do not lower release thresholds, claim the current candidate has
passed, or treat controller interception as improved raw model accuracy.
