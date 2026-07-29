# Correctness suite: single-head attention baseline

Official Spike/Verilator correctness workflow for the Gemmini transformer
decoder-block bareMetalC test. Calibrated Q/K scales, score dequant, and
RMSNorm gain are the **source defaults** — no opt-in calibration flag.

## What this suite validates

- Attention-stage numerical correctness: Q/K int8 quantization + score storage +
  softmax distribution vs an unclipped gold reference
- Residual / RMSNorm saturation safety (pre-clip |raw| band + no int8 sat)
- Final float-gold (Spike) or expected-snapshot (Verilator) match

**Validated scope:** single-head (`NUM_ATTENTION_HEADS=1`), `D_MODEL=16`,
`D_FF=64`, seeds `1–5`, `L∈{16,32,64,128,256}`.

## What it does *not* validate

- Multi-head attention (out of scope by design)
- Other shapes / widths beyond D16/F64
- FFN / output-projection stages as a standalone correctness claim
- RTL / silicon-level correctness beyond matching a Spike-exported snapshot
- Timing, power, or utilization regressions (see sweeps / baseline-tests)

## Headline results

(a) **RMSNorm gain retune** (`RMSNORM_GAIN=0.33974210`) eliminates pre-clip
saturation with zero correctness regressions — **25/25** gold PASS across the
validated grid; residual/RMSNorm peaks stay within the ~105–115 band.

(b) **int8 Q/K quantization** does not cause attention entropy collapse relative
to an unclipped gold reference — **0 rows flagged**, worst ΔH_norm **0.015**
(threshold 0.03); worst Δmax_w ≈ **−0.022** (threshold −0.03).

## How to run

One-command Spike gold grid (entry point):

```bash
./correctness/scripts/run_attention_baseline_grid.sh
# optional: ./correctness/scripts/run_attention_baseline_grid.sh 1 2   # seed subset
```

Export / install expected headers for Verilator (`seed=1`, all L):

```bash
./correctness/scripts/export_baseline_expected.sh
# or single L: ./correctness/scripts/spike_export_expected.sh 16 16 64 1
```

Verilator with a snapshot (no on-device gold):

```bash
./correctness/scripts/verilator_run_expected.sh 16 16 64 1
```

## What PASS means

A run prints `PASS` only if **all** of the following hold:

1. Final output matches float gold (Spike) or `expected_final` (USE_EXPECTED)
2. Residual 1/2 and RMSNorm 1/2: pre-clip max |raw| ≤ 115, and stored int8 does
   not hit `elem_t` min/max
3. Softmax fixed-vs-gold: max ΔH_norm ≤ 0.03 and min Δmax_w ≥ −0.03

Absolute H_norm rising with L is **length dilution**, not a failure signal
(printed only under `-DDBG_SOFTMAX_DIST=1` as informational).

## Interpreting failures

| Message | Meaning |
| --- | --- |
| `FAIL: N out-of-tolerance elements` | Final gold mismatch (numeric drift) |
| `FAIL: N elements differ from … expected_final` | Snapshot stale or RTL/software divergence |
| `FAIL residual/RMSNorm assertion: <stage> preclip…` | Pre-clip |raw| exceeded 115 |
| `FAIL residual/RMSNorm assertion: <stage> int8 saturation` | Hit ±127 on residual/RMSNorm |
| `FAIL softmax assertion: max delta_H_norm…` | Fixed softmax too uniform vs gold |
| `FAIL softmax assertion: min delta_max_w…` | Fixed row peak too diffuse vs gold |

## Single-head scope decision

`NUM_ATTENTION_HEADS=1` is intentional (laptop-scale testing), not incomplete
work. Extending to multi-head later would need per-head Q/K slices, softmax
probes per head, and a re-run of the seed×L grid — not assumed by this suite.

## Optional flags

| Flag | Role |
| --- | --- |
| `-DDBG_RESIDUAL_RMSNORM=1` | Verbose residual/RMSNorm pre-clip prints (assertions always on) |
| `-DDBG_SOFTMAX_DIST=1` | Verbose per-row softmax vs-gold prints + H_norm info |
| `-DUSE_LEGACY_QK_SCALES=1` | Old path: QUANT_SCALE for Q/K/score dequant, `RMSNORM_GAIN=0.4` (A/B only) |
| `-DPRNG_SEED`, `-DSKIP_GOLD`, `-DUSE_EXPECTED`, `-DDUMP_EXPECTED` | Unchanged |

Probe scripts (`probe_residual_rmsnorm.sh`, `probe_softmax_dist.sh`) remain for
verbose TSV collection only — prefer `run_attention_baseline_grid.sh` for CI.

## Artifacts

- `correctness/expected/L*_D16_F64_seed1.h` — official baseline snapshots
- `correctness/expected/include_*/transformer_expected.h` — include path for builds
- `correctness/expected/legacy_pre_cal/` — superseded pre-calibration headers
- `correctness/logs/` — Spike/Verilator / export logs

## Changelog

Calibrated ACC_SCALE_Q/K (`0.0059645441` / `0.0052780764`), SCORE_DEQUANT=`1/6`,
and RMSNORM_GAIN=`0.33974210` are now the default baseline in
`transformer_block_test.c` (replacing the old QUANT_SCALE / gain=0.4 path).
`USE_CALIBRATED_QK_SCALES` was replaced by `USE_LEGACY_QK_SCALES` (default off)
for A/B only. Residual/RMSNorm band checks and softmax fixed-vs-gold ΔH / Δmax_w
assertions always run; `DBG_*` flags are verbose-only. Expected headers dropped
the `_cal` suffix; pre-cal snapshots live under `expected/legacy_pre_cal/`.
Entry point: `run_attention_baseline_grid.sh`.
