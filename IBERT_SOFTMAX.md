# Hardware I-BERT Softmax (`GemminiIBertSoftmaxConfig`)

**Status:** Stock Gemmini I-BERT Softmax / Normalizer path enabled under
`has_normalizations` (separate from `GemminiRocketConfig` / custom PWL Softmax).  
**Companion:** Host Softmax error baseline remains [`SOFTMAX_ERROR_BASELINE.md`](SOFTMAX_ERROR_BASELINE.md);
stock-Gemmini cycle baseline remains [`BASELINE.md`](BASELINE.md);
HW-PWL Softmax cycles remain [`PWL_SOFTMAX.md`](PWL_SOFTMAX.md).  
**RTL / SW:** Existing `Normalizer.scala` + `AccumulatorScale` `iexp` (restored /
enabled — not new RTL), `tiled_norm_auto(..., SOFTMAX, …)` in `gemmini.h`,
Spike `apply_norm` / `apply_iexp`, header `gemmini_params_ibert.h`
(`HAS_NORMALIZATIONS`).  
**Test flag:** `-DUSE_HW_SOFTMAX=1`.

---

## Implementation context

I-BERT Softmax is Gemmini’s **pre-existing** (previously disabled) hardware
path — Normalizer + integer `iexp` in AccumulatorScale — restored and wired
through `GemminiIBertSoftmaxConfig` / `USE_HW_SOFTMAX`, not built from scratch.
PWL Softmax ([`PWL_SOFTMAX.md`](PWL_SOFTMAX.md)) is a **from-scratch** Chisel
unit (`PwlSoftmax` + bridge). That asymmetry is why PWL remains the primary
deliverable even though the I-BERT path “works” on Spike and Verilator: I-BERT
is a reference enablement of upstream Gemmini Softmax HW; PWL is the
project-owned Softmax design.

---

## Spike correctness (post-`iexp` restore)

| Suite | Grid | Result |
| --- | --- | --- |
| Random | `L ∈ {16, 32, 64, 128, 256}` × seeds 1–5 | **25 / 25 PASS** |
| Edges | `all_ones`, `all_max_mag`, `one_hot`, `checkerboard` × same L × seeds | **100 / 100 PASS** |

Path: `USE_HW_SOFTMAX` → `tiled_norm_auto(..., SOFTMAX, SCORE_DEQUANT)`.
Artifact: `chipyard/tmp_baseline_validation/softmax_err_char_hw_post_iexp/`.

---

## Verilator (`GemminiIBertSoftmaxConfig`, seed=1)

`D_MODEL=16`, `D_FF=64`. Wall time is host Verilator sim time only — use cycle
columns for architecture. Source:
`chipyard/tmp_baseline_validation/phase3_verilator_hw_ibert/report.tsv`
(and `SUMMARY.txt`). Harness:
`chipyard/tmp_baseline_validation/run_phase3_verilator_hw_ibert.sh`.

**Scoped run:** `BREAK_SIM_PREREQ=1` against the already-built
`simulator-chipyard.harness-GemminiIBertSoftmaxConfig`; no unscoped
`clean` / `clean-all-configs` (Rocket / PWL simulator binaries left intact).
L=16 run first as smoke; then L=32/64/128. L=256 skipped (wall-clock cost).

| L | Result | Total cycles | Softmax cycles | Wall (s) | Softmax share |
| ---: | --- | ---: | ---: | ---: | ---: |
| 16 | **PASS** | 55998 | 2437 | 1155 | **4.4%** |
| 32 | **PASS** | 110496 | 9175 | 2305 | **8.3%** |
| 64 | **PASS** | 227311 | 29513 | 6206 | **13.0%** |
| 128 | **PASS** | 528387 | 131411 | 22559 | **24.9%** |
| 256 | skipped | — | — | — | — |

Per-stage cycles from the same Verilator logs (`print_stage_report` /
`* cycles:` lines — same instrumentation as the scalar baseline and
[`PWL_SOFTMAX.md`](PWL_SOFTMAX.md)). Exact values from `report.tsv` (not
re-derived):

| L | QKV | Scores | Softmax | Attn×V | Out proj | Res1 | RMSNorm1 | FFN | Res2 | RMSNorm2 | **Total** | Wall (s) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 2537 | 739 | 2437 | 712 | 710 | 8370 | 15202 | 1955 | 8254 | 15082 | **55998** | 1155 |
| 32 | 2887 | 1090 | 9175 | 1060 | 828 | 16478 | 30065 | 2641 | 16344 | 29928 | **110496** | 2305 |
| 64 | 3680 | 2479 | 29513 | 1567 | 1077 | 32762 | 60068 | 3909 | 32455 | 59801 | **227311** | 6206 |
| 128 | 6382 | 8221 | 131411 | 3836 | 1807 | 65217 | 119610 | 7850 | 64712 | 119341 | **528387** | 22559 |

No DMA or other hardware asserts on any L. No Softmax mismatch / hang /
unexpected cycle-count anomaly beyond the known L=128 Residual class already
documented for PWL (below).

### Softmax “utilization” (same metric gap as PWL)

Softmax % above is **Softmax_cycles ÷ total** from the stage timer — **not**
measured functional-unit busy%. Softmax is not a GEMM stage, so it has no
`RUN_GEMMINI_MATMUL` harness counters (`exe_busy` / `mesh_util` apply only to
QKV / Scores / Attn×V / Out / FFN). There are also **no** busy-cycle counters
for Normalizer / AccumulationLanes / MaxLanes / AccumulatorScale in this
harness — the same limitation as PWL Softmax FU utilization in
[`PWL_SOFTMAX.md`](PWL_SOFTMAX.md). Do not read Softmax share as Softmax HW
utilization.

### Artifacts

`chipyard/tmp_baseline_validation/phase3_verilator_hw_ibert/` —
`report.tsv`, `SUMMARY.txt`, `logs/L*_D16_F64_seed1.verilator.log`, binaries,
Spike expected exports.

Harness: `chipyard/tmp_baseline_validation/run_phase3_verilator_hw_ibert.sh`.

---

## Comparison vs. PWL Softmax and Scalar Baseline

Side-by-side with Verilator numbers already published in
[`BASELINE.md`](BASELINE.md) (§2, `GemminiRocketConfig`, seed=1, HW-resadd
default-on) and [`PWL_SOFTMAX.md`](PWL_SOFTMAX.md) (post-tiling-fix
`GemminiPWLSoftmaxConfig`, seed=1). Those docs are not modified; this section
is additive only.

### Comparability

| Item | Scalar (`BASELINE.md`) | HW-PWL (`PWL_SOFTMAX.md`) | HW-IBert (this doc) |
| --- | --- | --- | --- |
| `D_MODEL` / `D_FF` / `DIM` | 16 / 64 / 16 | 16 / 64 / 16 | 16 / 64 / 16 |
| `PRNG_SEED` / case | 1 / random | 1 / random | 1 / random |
| `USE_HW_RESADD` | default **1** | default **1** | default **1** |
| `QUANT_SCALE` / final tol | `5 × QUANT_SCALE` | same | same |
| Softmax implementation | Host scalar (`expf` + largest-remainder) | HW PWL (`USE_HW_PWL_SOFTMAX=1`) | HW I-BERT (`USE_HW_SOFTMAX=1`) |
| Chipyard config | `GemminiRocketConfig` | `GemminiPWLSoftmaxConfig` | `GemminiIBertSoftmaxConfig` |

### Softmax cycle counts (Verilator, seed=1)

Scalar Softmax from [`BASELINE.md`](BASELINE.md); HW-PWL Softmax from
[`PWL_SOFTMAX.md`](PWL_SOFTMAX.md); HW-IBert Softmax from `report.tsv` above.
Softmax % is Softmax cycles ÷ total for that build. Scalar Softmax dominates
the scalar baseline (**67.2% → 85.9% → 95.1% → 98.1%** of total as L grows).

| L | Scalar Softmax | HW-PWL Softmax | HW-IBert Softmax | PWL ÷ IBert (×) | Scalar Softmax % | PWL Softmax % | IBert Softmax % |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 109903 | 3159 | 2437 | 1.296 | 67.2% | 5.6% | 4.4% |
| 32 | 619397 | 13268 | 9175 | 1.446 | 85.9% | 11.6% | 8.3% |
| 64 | 3831599 | 47958 | 29513 | 1.625 | 95.1% | 19.5% | 13.0% |
| 128 | 21070239 | 227085 | 131411 | 1.728 | 98.1% | 36.4% | 24.9% |

Both hardware Softmax paths are **dramatically faster** than host scalar Softmax
(tens of × at L=16, approaching ~100× Softmax-cycle reduction at L=128). Between
the two HW paths, **I-BERT Softmax uses fewer absolute Softmax cycles than PWL**
at every L (~**1.3–1.7×** fewer Softmax cycles; PWL Softmax ÷ IBert Softmax in
the table). Totals follow the same ordering (IBert total &lt; PWL total &lt;&lt;
scalar total) because Softmax is the stage that moved.

### Accuracy tradeoff (not a pass/fail difference)

From prior Softmax error characterization (Spike random grids; Step-3 /
`CHAR_SOFTMAX_ERR` work under `chipyard/tmp_baseline_validation/`, summarized
alongside [`SOFTMAX_ERROR_BASELINE.md`](SOFTMAX_ERROR_BASELINE.md)):

| Path | `soft_max_ulps` range | Accuracy (best → worst) |
| --- | --- | --- |
| Scalar host | **[0.608, 0.696]** | best |
| HW PWL | **[1.638, 5.044]** | middle |
| HW I-BERT | **[2.888, 18.741]** | worst of the three |

Ordering: **scalar &lt; PWL &lt; HW I-BERT** (best to worst per-element Softmax
error). All three stay well under the final `5 × QUANT_SCALE` tolerance
(`final_max_over_tol` peaks remain &lt; ~0.5). Frame this as a **speed /
accuracy tradeoff between the two hardware Softmax paths** (I-BERT: fewer
Softmax cycles, higher Softmax ulps; PWL: more Softmax cycles than I-BERT,
tighter Softmax error) — not a PASS/FAIL difference. That tradeoff, plus
PWL being project-owned RTL, is why PWL is pursued as the primary Softmax
deliverable even though I-BERT Verilator PASS is real.

### Non-Softmax stage parity (IBert vs PWL)

At L=16/32/64, QKV / Scores / Attn×V / Out / FFN / Residual / RMSNorm absolute
cycles match [`PWL_SOFTMAX.md`](PWL_SOFTMAX.md) within noise (typically tens of
cycles). The two configs are apples-to-apples outside Softmax itself; Softmax
is the intentional RTL difference (`has_normalizations` vs `has_pwl_softmax`).

### L=128 Residual anomaly (same class as PWL vs scalar)

| Build | Res1 | Res2 |
| --- | ---: | ---: |
| HW-IBert (this sweep) | **65217** | **64712** |
| HW-PWL ([`PWL_SOFTMAX.md`](PWL_SOFTMAX.md)) | 65301 | 64664 |
| Scalar ([`BASELINE.md`](BASELINE.md)) | 73609 | 72927 |

IBert Res1/Res2 match PWL within noise (deltas −84 / +48), and both sit
~**11%** below scalar. This is the **same already-diagnosed anomaly class**
documented in [`PWL_SOFTMAX.md`](PWL_SOFTMAX.md) (L=128 Residual investigation) —
**not** a new I-BERT-specific bug. Do not re-investigate here; treat L=128
Residual absolute counts as non-comparable to scalar when reading share
shifts. Softmax and total-cycle columns remain the measured values from each
published run.

---

## Related

| Doc / path | Role |
| --- | --- |
| [`BASELINE.md`](BASELINE.md) | Stock Gemmini (host Softmax) cycle baseline |
| [`PWL_SOFTMAX.md`](PWL_SOFTMAX.md) | Custom HW-PWL Softmax Verilator / Spike validation |
| [`SOFTMAX_ERROR_BASELINE.md`](SOFTMAX_ERROR_BASELINE.md) | Step-1 float→int8 Softmax quantization error (scalar) |
| `chipyard/tmp_baseline_validation/softmax_err_char_hw_post_iexp/` | Spike I-BERT Softmax error char (post-`iexp`) |
| `chipyard/tmp_baseline_validation/phase3_verilator_hw_ibert/` | This Verilator L-sweep (`report.tsv`, logs) |
