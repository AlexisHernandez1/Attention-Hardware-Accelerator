# Hardware I-BERT Softmax (`GemminiIBertSoftmaxConfig`)

**Status:** Stock Gemmini I-BERT Softmax / Normalizer path enabled under
`has_normalizations` (separate from `GemminiRocketConfig` / custom PWL Softmax).  
**Companion:** Host Softmax error baseline remains [`SOFTMAX_ERROR_BASELINE.md`](SOFTMAX_ERROR_BASELINE.md);
stock-Gemmini cycle baseline remains [`BASELINE.md`](BASELINE.md);
HW-PWL Softmax cycles remain [`PWL_SOFTMAX.md`](PWL_SOFTMAX.md);
cross-build comparisons remain [`COMPARISON.md`](COMPARISON.md).  
**RTL / SW:** Existing `Normalizer.scala` + `AccumulatorScale` `iexp` (restored /
enabled — not new RTL), `tiled_norm_auto(..., SOFTMAX, …)` in `gemmini.h`,
Spike `apply_norm` / `apply_iexp`, header `gemmini_params_ibert.h`
(`HAS_NORMALIZATIONS`).  
**Test flag:** `-DUSE_HW_SOFTMAX=1`.

---

## Spike correctness (post-`iexp` restore)

| Suite | Grid | Result |
| --- | --- | --- |
| Random | `L ∈ {16, 32, 64, 128, 256}` × seeds 1–5 | **25 / 25 PASS** |
| Edges | `all_ones`, `all_max_mag`, `one_hot`, `checkerboard` × same L × seeds | **100 / 100 PASS** |

Path: `USE_HW_SOFTMAX` → `tiled_norm_auto(..., SOFTMAX, SCORE_DEQUANT)`.
Artifact: `chipyard/tmp_baseline_validation/softmax_err_char_hw_post_iexp/`.

### Softmax error (this build only)

From `chipyard/tmp_baseline_validation/softmax_err_char_hw_post_iexp/`:
`soft_max_ulps` range **[2.888, 18.741]** on the random Spike grid (Part A).
For cross-path accuracy ordering vs scalar / PWL, see [`COMPARISON.md`](COMPARISON.md).

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
`* cycles:` lines — same instrumentation as the scalar baseline). Exact values
from `report.tsv` (not re-derived):

| L | QKV | Scores | Softmax | Attn×V | Out proj | Res1 | RMSNorm1 | FFN | Res2 | RMSNorm2 | **Total** | Wall (s) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 2537 | 739 | 2437 | 712 | 710 | 8370 | 15202 | 1955 | 8254 | 15082 | **55998** | 1155 |
| 32 | 2887 | 1090 | 9175 | 1060 | 828 | 16478 | 30065 | 2641 | 16344 | 29928 | **110496** | 2305 |
| 64 | 3680 | 2479 | 29513 | 1567 | 1077 | 32762 | 60068 | 3909 | 32455 | 59801 | **227311** | 6206 |
| 128 | 6382 | 8221 | 131411 | 3836 | 1807 | 65217 | 119610 | 7850 | 64712 | 119341 | **528387** | 22559 |

No DMA or other hardware asserts on any L. No Softmax mismatch / hang /
unexpected cycle-count anomaly on this sweep. For the L=128 Residual class
shared with PWL vs scalar, see [`COMPARISON.md`](COMPARISON.md).

### Softmax “utilization”

Softmax % above is **Softmax_cycles ÷ total** from the stage timer — **not**
measured functional-unit busy%. Softmax is not a GEMM stage, so it has no
`RUN_GEMMINI_MATMUL` harness counters (`exe_busy` / `mesh_util` apply only to
QKV / Scores / Attn×V / Out / FFN). There are also **no** busy-cycle counters
for Normalizer / AccumulationLanes / MaxLanes / AccumulatorScale in this
harness. Do not read Softmax share as Softmax HW utilization.

### Artifacts

`chipyard/tmp_baseline_validation/phase3_verilator_hw_ibert/` —
`report.tsv`, `SUMMARY.txt`, `logs/L*_D16_F64_seed1.verilator.log`, binaries,
Spike expected exports.

Harness: `chipyard/tmp_baseline_validation/run_phase3_verilator_hw_ibert.sh`.

For cross-build comparisons and analysis, see [`COMPARISON.md`](COMPARISON.md).

---

## Related

| Doc / path | Role |
| --- | --- |
| [`BASELINE.md`](BASELINE.md) | Stock Gemmini (host Softmax) cycle baseline |
| [`PWL_SOFTMAX.md`](PWL_SOFTMAX.md) | Custom HW-PWL Softmax Verilator / Spike validation |
| [`COMPARISON.md`](COMPARISON.md) | Cross-build Softmax cycle / accuracy / residual analysis |
| [`SOFTMAX_ERROR_BASELINE.md`](SOFTMAX_ERROR_BASELINE.md) | Step-1 float→int8 Softmax quantization error (scalar) |
| `chipyard/tmp_baseline_validation/softmax_err_char_hw_post_iexp/` | Spike I-BERT Softmax error char (post-`iexp`) |
| `chipyard/tmp_baseline_validation/phase3_verilator_hw_ibert/` | This Verilator L-sweep (`report.tsv`, logs) |
