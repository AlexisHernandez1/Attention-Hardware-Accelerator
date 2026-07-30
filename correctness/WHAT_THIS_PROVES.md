# What this proves

Framed around attention-stage hardware / software correctness for the Gemmini
single-head decoder-block baseline (random PRNG grid **and** named edge-case
inputs).

## Claim → evidence → scope

### Claim 1: RMSNorm gain retune keeps residual/RMSNorm in int8 safely

- **Claim:** With calibrated softmax magnitudes, lowering `RMSNORM_GAIN` to
  `0.33974210` removes residual/RMSNorm pre-clip saturation without breaking
  float-gold agreement.
- **Evidence:** Random baseline seeds 1–5 × L∈{16…256}: **25/25 gold PASS**,
  **0/25** any-stage saturation; post-retune peaks ≈ rms1_out 110, res2 112,
  rms2_out 111 (within the ~105–115 band). Enforced as always-on assertions
  (`preclip ≤ 115`, no `elem_t` min/max on Residual/RMSNorm 1–2) for the random
  case.
- **Scope:** Single-head, D16/F64, L≤256, seeds 1–5. Does not prove other gains,
  multi-head, or FFN-stage saturation independently.

### Claim 2: Hardware Q/K int8 quantization preserves attention distribution

- **Claim:** Scoring softmax on int8-stored Q@Kᵀ (with SCORE_DEQUANT=1/6) does
  not collapse / inflate row entropy relative to an unclipped float gold built
  from the same int8 Q/K before score round/clip.
- **Evidence:** Gold-comparison probe across the same 25 random configs: **0
  rows** above ΔH_norm threshold; worst ΔH_norm **0.015 < 0.03**; worst Δmax_w ≈
  **−0.022 ≥ −0.03**. Absolute H_norm rising with L is length dilution, not a
  failure. Assertions always on for `EDGE_CASE_ID=0`.
- **Scope:** Single-head (`layer=0`, `head=0`), D16/F64, seeds 1–5, L≤256.
  Not yet shown for multi-head or other shapes.

### Claim 3: End-to-end block still matches gold under calibrated defaults

- **Claim:** With the official calibrated scales as defaults, Spike float-gold
  (and Spike-exported expected snapshots for Verilator) still PASS.
- **Evidence:** `run_attention_baseline_grid.sh` random rows; expected headers
  under `correctness/expected/L*_D16_F64_seed1.h`.
- **Scope:** Same L×seed grid; Verilator expected mode checks snapshot match
  only (does not re-roll gold).

### Claim 4: Named edge-case inputs preserve safety / distribution contracts

- **Claim:** The seven adversarial / degenerate input fills below pass the
  same L×seed grid with case-specific assertions (clamp-to-bound where sat is
  intentional; softmax / sign / underflow checks where relevant).
- **Evidence:** Unified default grid includes all seven: **175/175** edge rows
  PASS (with Claim 3’s **25/25** random rows → **200/200** total).
- **Scope:** Single-head, D16/F64, seeds 1–5, **L≤256 only** (not validated
  beyond). `all_negative` at L=256 is in-scope after the softmax mass fix
  below; L=512+ is explicitly out of scope for this suite.

| ID | Name | What it checks |
| --- | --- | --- |
| 1 | `all_zeros` | Q/K/V=0; RMSNorm ε; softmax → uniform, no NaN |
| 2 | `all_ones` | Max int8 X; ACC_SCALE clamp (no wrap); int32 acc in range |
| 3 | `all_max_mag` | ±128/127 pattern; clamp-to-bound instead of “no sat” |
| 4 | `one_hot` | Softmax collapses; Δmax_w at extreme |
| 5 | `checkerboard` | Adversarial variance; RMSNorm preclip ≤115 (rail X residuals clamp-checked) |
| 6 | `all_negative` | Sign handling in quantized Q/K/V; Residual-1 band through L=256 |
| 7 | `near_zero` | RMSNORM_GAIN must not underflow final output to all zeros |

### Softmax weight mass (permanent quantization fix)

- **Claim:** Int8 softmax weights must preserve Σw ≈ 1.0 under `QUANT_SCALE`
  (1/128). Independent round-half-up does not: at L=256 uniform attention,
  `128/L=0.5` rounds to 1 for every key → Σw_dequant≈2 and Residual-1 exceeds
  the calibrated band.
- **Evidence / fix:** Hamilton / largest-remainder renormalization in
  `quantized_softmax` (target raw sum = 128). No-op for L∈{16,32,64,128} where
  `128/L` is already integral. Post-fix `all_negative` L=256: attn max\|raw\|
  128→64, Residual-1 peaks back under 115 (e.g. seed1 126→95); full edge grid
  171/175 → 175/175.
- **Scope:** Softmax weight rounding only — **not** a recalibration of
  `ACC_SCALE_Q/K`, `SCORE_DEQUANT`, or `RMSNORM_GAIN`. Validated through L=256.

## Out of scope (explicit)

Multi-head attention, non-D16/F64 shapes, L>256, dedicated FFN correctness
claims, timing/power, and silicon bring-up. Extend only with a new grid and
probes — do not silently generalize these results.

## Changelog (edge promotion + softmax mass)

- **Softmax mass fix (Jul 2026):** Replaced per-weight round-half-up of softmax
  probs with largest-remainder so each row’s int8 weights sum to 128. Root
  cause found via `DBG_RESIDUAL_TRACE` on `all_negative` @ L=256 (archived under
  `correctness/archive/diagnostics/`).
- **Edge cases promoted to default baseline:** Removed `EDGE_CASE_SWEEP`.
  `EDGE_CASE_ID=0` is random PRNG; `1..7` are named fills. Default
  `run_attention_baseline_grid.sh` runs random + all seven edges (no `--edge`).
