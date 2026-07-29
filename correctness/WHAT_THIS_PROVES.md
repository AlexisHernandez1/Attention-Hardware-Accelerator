# What this proves

Framed around attention-stage hardware / software correctness for the Gemmini
single-head decoder-block baseline.

## Claim → evidence → scope

### Claim 1: RMSNorm gain retune keeps residual/RMSNorm in int8 safely

- **Claim:** With calibrated softmax magnitudes, lowering `RMSNORM_GAIN` to
  `0.33974210` removes residual/RMSNorm pre-clip saturation without breaking
  float-gold agreement.
- **Evidence:** Full grid seeds 1–5 × L∈{16…256}: **25/25 gold PASS**, **0/25**
  any-stage saturation; post-retune peaks ≈ rms1_out 110, res2 112, rms2_out 111
  (within the ~105–115 band). Now enforced as always-on assertions
  (`preclip ≤ 115`, no `elem_t` min/max on Residual/RMSNorm 1–2).
- **Scope:** Single-head, D16/F64, L≤256, seeds 1–5. Does not prove other gains,
  multi-head, or FFN-stage saturation independently.

### Claim 2: Hardware Q/K int8 quantization preserves attention distribution

- **Claim:** Scoring softmax on int8-stored Q@Kᵀ (with SCORE_DEQUANT=1/6) does
  not collapse / inflate row entropy relative to an unclipped float gold built
  from the same int8 Q/K before score round/clip.
- **Evidence:** Gold-comparison probe across the same 25 configs: **0 rows**
  above ΔH_norm threshold; worst ΔH_norm **0.015 < 0.03**; worst Δmax_w ≈
  **−0.022 ≥ −0.03**. Absolute H_norm rising with L is length dilution, not a
  failure. Assertions are always on in the baseline binary.
- **Scope:** Single-head (`layer=0`, `head=0`), D16/F64, seeds 1–5, L≤256.
  Not yet shown for multi-head or other shapes.

### Claim 3: End-to-end block still matches gold under calibrated defaults

- **Claim:** With the official calibrated scales as defaults, Spike float-gold
  (and Spike-exported expected snapshots for Verilator) still PASS.
- **Evidence:** `run_attention_baseline_grid.sh` / prior calibrated full-grid
  validation; expected headers under `correctness/expected/L*_D16_F64_seed1.h`.
- **Scope:** Same grid; Verilator expected mode checks snapshot match only
  (does not re-roll gold).

## Out of scope (explicit)

Multi-head attention, non-D16/F64 shapes, dedicated FFN correctness claims,
timing/power, and silicon bring-up. Extend only with a new grid and probes — do
not silently generalize these results.
