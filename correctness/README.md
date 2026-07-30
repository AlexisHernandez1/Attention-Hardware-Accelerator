# Correctness suite: single-head attention baseline

Official Spike/Verilator correctness workflow for the Gemmini transformer
decoder-block bareMetalC test. Calibrated Q/K scales, score dequant, RMSNorm
gain, and **largest-remainder softmax weight quantization** are the source
defaults. The default grid covers the random PRNG baseline **and** seven named
edge-case input fills.

## What this suite validates

- Attention-stage numerical correctness: Q/K int8 quantization + score storage +
  softmax distribution vs an unclipped gold reference (random baseline)
- Softmax int8 weight mass: each row’s weights sum to 128 under `QUANT_SCALE`
  (largest-remainder; prevents L=256 uniform mass-doubling)
- Residual / RMSNorm saturation safety (pre-clip |raw| band + no int8 sat, or
  clamp-to-bound on intentional saturation edges)
- Named edge inputs: zeros, ones, max-mag, one-hot, checkerboard, all-negative,
  near-zero (see `WHAT_THIS_PROVES.md`)
- Final float-gold (Spike, random case) or expected-snapshot (Verilator) match

**Validated scope:** single-head (`NUM_ATTENTION_HEADS=1`), `D_MODEL=16`,
`D_FF=64`, seeds `1–5`, `L∈{16,32,64,128,256}`.

## What it does *not* validate

- Multi-head attention (out of scope by design)
- Other shapes / widths beyond D16/F64, or L>256
- FFN / output-projection stages as a standalone correctness claim
- RTL / silicon-level correctness beyond matching a Spike-exported snapshot
- Timing, power, or utilization regressions (see sweeps / baseline-tests)

## Headline results

(a) **RMSNorm gain retune** (`RMSNORM_GAIN=0.33974210`) eliminates pre-clip
saturation with zero correctness regressions — **25/25** random-grid gold PASS;
residual/RMSNorm peaks stay within the ~105–115 band.

(b) **int8 Q/K quantization** does not cause attention entropy collapse relative
to an unclipped gold reference — **0 rows flagged**, worst ΔH_norm **0.015**
(threshold 0.03); worst Δmax_w ≈ **−0.022** (threshold −0.03).

(c) **Edge-case grid** (7 fills × same L×seed) — **175/175 PASS** after softmax
mass fix; combined default grid **200/200**.

## Defaults (source)

| Constant | Default | Role |
| --- | --- | --- |
| `ACC_SCALE_Q` | `0.0059645441` | Q projection ACC → ~108 int8 headroom |
| `ACC_SCALE_K` | `0.0052780764` | K projection ACC → ~108 int8 headroom |
| `SCORE_DEQUANT_SCALE` | `1/6` | Softmax logit scale for stored scores |
| `RMSNORM_GAIN` | `0.33974210` | Keeps residual/RMSNorm preclip in band |
| Softmax weight quant | largest-remainder → Σraw=128 | Preserves Σw≈1.0 (fixes L=256 uniform) |

`USE_LEGACY_QK_SCALES=1` restores QUANT_SCALE / gain=0.4 for A/B only.

## How to run

One-command Spike grid (random baseline **+** all 7 edge cases):

```bash
./correctness/scripts/run_attention_baseline_grid.sh
# optional seed subset: ./correctness/scripts/run_attention_baseline_grid.sh 1 2
# resume after interrupt: ./correctness/scripts/run_attention_baseline_grid.sh --resume
```

Summary TSV columns: `case`, `L`, `seed`, `result` (per-case totals printed at
the end). Logs: `correctness/logs/baseline_grid/`.

Export / install expected headers for Verilator (`seed=1`, all L, random case):

```bash
./correctness/scripts/export_baseline_expected.sh
# or single L: ./correctness/scripts/spike_export_expected.sh 16 16 64 1
```

Verilator with a snapshot (no on-device gold):

```bash
./correctness/scripts/verilator_run_expected.sh 16 16 64 1
```

## What PASS means

**Random baseline (`EDGE_CASE_ID=0`):**

1. Final output matches float gold (Spike) or `expected_final` (USE_EXPECTED)
2. Residual/RMSNorm: pre-clip max |raw| ≤ 115, no int8 sat
3. Softmax fixed-vs-gold: max ΔH_norm ≤ 0.03 and min Δmax_w ≥ −0.03

**Edge cases (`EDGE_CASE_ID=1..7`):** case-specific assertions (see
`WHAT_THIS_PROVES.md`); residual/RMSNorm band or clamp rules as applicable.
Final float-gold is not the pass criterion for edge fills.

## Interpreting failures

| Message | Meaning |
| --- | --- |
| `FAIL: N out-of-tolerance elements` | Final gold mismatch (random case) |
| `FAIL: N elements differ from … expected_final` | Snapshot stale or RTL/software divergence |
| `FAIL residual/RMSNorm assertion: <stage> preclip…` | Pre-clip |raw| exceeded 115 |
| `FAIL residual/RMSNorm assertion: <stage> int8 saturation` | Hit ±127 where sat is forbidden |
| `FAIL residual/RMSNorm assertion: <stage> clamp wrap…` | Exceeded range but did not clamp to ±bound |
| `FAIL softmax assertion: max delta_H_norm…` | Fixed softmax too uniform vs gold |
| `FAIL edge …` | Named edge-case assertion (see log for which) |

## Single-head scope decision

`NUM_ATTENTION_HEADS=1` is intentional (laptop-scale testing), not incomplete
work. Extending to multi-head later would need per-head Q/K slices, softmax
probes per head, and a re-run of the seed×L×case grid.

## Optional flags

| Flag | Role |
| --- | --- |
| `-DEDGE_CASE_ID=<0..7>` | Input fill (grid sets this; 0=random default) |
| `-DDBG_RESIDUAL_RMSNORM=1` | Verbose residual/RMSNorm pre-clip prints |
| `-DDBG_SOFTMAX_DIST=1` | Verbose per-row softmax vs-gold prints + H_norm info |
| `-DDBG_RESIDUAL_TRACE=1` | Diagnostic residual/attn growth traces (see archive) |
| `-DUSE_LEGACY_QK_SCALES=1` | Old QUANT_SCALE / gain=0.4 path (A/B only) |
| `-DPRNG_SEED`, `-DSKIP_GOLD`, `-DUSE_EXPECTED`, `-DDUMP_EXPECTED` | Unchanged |

## Artifacts

- `correctness/expected/L*_D16_F64_seed1.h` — official baseline snapshots
- `correctness/expected/include_*/transformer_expected.h` — include path for builds
- `correctness/expected/legacy_pre_cal/` — superseded pre-calibration headers
- `correctness/logs/baseline_grid/` — default grid Spike logs + `summary.tsv`
- `correctness/archive/diagnostics/` — archived all_negative residual-trace tooling

## Changelog

Calibrated ACC_SCALE_Q/K, SCORE_DEQUANT=`1/6`, RMSNORM_GAIN=`0.33974210`, and
largest-remainder softmax weight quantization are the defaults.
`EDGE_CASE_SWEEP` was removed; edge fills are part of
`run_attention_baseline_grid.sh` (no `--edge`). See `WHAT_THIS_PROVES.md` for
claim/evidence and the Jul 2026 softmax-mass fix note.
