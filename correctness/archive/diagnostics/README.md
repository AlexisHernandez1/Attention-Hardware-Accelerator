# Archived diagnostics (all_negative Residual-1 @ L=256)

One-off investigation that found int8 softmax weight mass doubling at L=256
(`round(128/L)=1` for every key → Σw_dequant≈2). Fixed permanently in
`quantized_softmax` via Hamilton / largest-remainder renormalization
(target sum = 128). Kept here for historical root-cause replay.

## Contents

- `trace_all_negative_residual.sh` — builds with `-DDBG_RESIDUAL_TRACE=1`
  and `-DEDGE_CASE_ID=6` (all_negative); writes per-L traces.
- `residual_trace_all_negative/` — Spike/trace logs from the investigation.
- `softmax_mass_fix/` — before/after L=256 seed checks after the fix.
- `edge_case_grid/` — earlier flag-gated edge-only sweep logs (superseded by
  the unified `correctness/logs/baseline_grid/` once edge cases joined the
  default grid).

Not part of the default CI entry point. Re-run only if debugging a similar
softmax-mass / residual-band issue.
