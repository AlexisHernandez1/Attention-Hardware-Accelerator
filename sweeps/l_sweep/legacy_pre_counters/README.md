# Legacy pre-counters-inline L=256 Verilator artifacts

**Different build — not the same series as the current L=16–128 baseline.**

These files are from a **materially different** build of `transformer_block_test`
than the counters-inline / deferred-printf baseline reported in
`../README.md` (L=16 / 32 / 64 / 128). They are kept here as a **historical
archive only**. Do **not** merge them into that table, and do **not** plot or
cite their cycle counts alongside the current baseline as one continuous scaling
series.

## Build mismatch (read this first)

### 1. GEMM stage timing was measured differently

This archived run used the old `run_gemmini_matmul_hw_util` path: counter
reset/read **plus a `printf` inside each GEMM stage’s timed window**.

The current L=16–128 baseline uses the permanent counters-inline /
deferred-printf path: counters still wrap each matmul, but **no `printf`
inside any timed region** (utilization is batched after all stage timers close).

So GEMM-stage cycle numbers between the two are **not comparable**. The
archived run’s GEMM stages include host I/O overhead that the current baseline’s
do not. Example from this log: QKV wall shown as **58,543** cycles — that figure
is inflated by in-window util printf and must not be lined up with current-baseline
QKV walls (~2.5k–6k at L=16–128).

### 2. Broader code / calibration state may also differ

This run predates the counters-inline permanent change. Depending on exactly when
it was generated, it **may also predate** later calibration / scale work (e.g.
`SCORE_DEQUANT_SCALE`, the ACC_SCALE robust-calibration pass, RMSNorm gain
freeze, etc.). Do **not** assume it matches today’s calibrated defaults without
checking commit history / log timestamps against the calibration timeline. This
README does **not** assert a specific calibration state.

### Bottom line

The **127,162,675** total cycle count and **125,890 s** (~35 h) wall time from
this archive **must not** be plotted, tabled, or cited alongside the current
L=16 / 32 / 64 / 128 baseline numbers as if they belong to the same series.

If referenced at all, treat them as a **separate, differently-conditioned** data
point — e.g. “an earlier build completed a full L=256 Verilator run in ~35 hours
without hitting the simulation timeout” — **not** as a cycle count on the current
scaling trend.

## What the logs contain (structural check; not independently re-verified)

From `logs/L256_D16_F64_seed1.verilator.log` and the wrapper `verilator_L256.log`:

- A `PASS` marker is present.
- A final timing block is present (`Transformer decoder-block timing` through
  `Total cycles: 127162675 (100%)`), plus a traffic estimate and harness
  `$finish` / make exit — i.e. the log is **not** truncated mid-Softmax the way
  the later incomplete counters-inline L=256 attempt was.
- Wrapper reports `wall_clock_seconds: 125890` and `hit_timeout: no`.
- Per-GEMM lines are the older inline `GEMM HW util …` printf style (see build
  mismatch above).

**This archive has not been independently re-verified** against a fresh
counters-inline rebuild. Structural completeness ≠ validated for the current
baseline.

## Contents

| File | Role |
| --- | --- |
| `verilator_L256.log` | Summary wrapper written by the unattended sweep harness |
| `logs/L256_D16_F64_seed1.verilator.log` | Raw Verilator/make stdout |
| `logs/L256_D16_F64_seed1.verilator_seconds` | Recorded wall seconds |
| `nohup_launch.log` | Unattended launcher stdout (includes L=256 stanza) |
| `verilator_seed1_unattended_stdout.log` | Same sweep stdout capture |

`MANIFEST.txt` is only a file listing (no completeness / comparability claims).

## Relation to current baseline

- Current validated Verilator table: L=16, 32, 64, 128 only (`../README.md`).
- Counters-inline L=256 was **not** completed (incomplete mid-Softmax run purged).
- Spike L=256 (full correctness grid) is unaffected and lives under
  `correctness/logs/`, not here.
