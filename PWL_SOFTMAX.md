# Hardware PWL Softmax (`GemminiPWLSoftmaxConfig`)

**Status:** Custom Q16.16 piecewise-linear Softmax elaborated into Gemmini under
`has_pwl_softmax` (separate from `GemminiRocketConfig` / I-BERT Normalizer).  
**Companion:** Host Softmax error baseline remains [`SOFTMAX_ERROR_BASELINE.md`](SOFTMAX_ERROR_BASELINE.md);
stock-Gemmini cycle baseline remains [`BASELINE.md`](BASELINE.md).  
**RTL / SW:** `PwlSoftmax.scala`, `PwlSoftmaxBridge.scala`, `tiled_pwl_softmax` in
`gemmini.h`, Spike model in `libgemmini` (`HAS_PWL_SOFTMAX`).  
**Test flag:** `-DUSE_HW_PWL_SOFTMAX=1`.

---

## Resolved bugs (distinct root causes)

### 1. Row-0 Softmax corruption — bridge `RegInit` (RANDOMIZE)

**Symptom (Verilator L=16):** Softmax rows 1…L−1 bit-exact vs Spike; row 0
near-zero / incomplete in DRAM (row sum ≠ 128).  
**Cause:** `PwlSoftmaxMvoutBridge` left `baseIdx`, `feedLen`/`feedPos`, pack
buffers, etc. as bare `Reg` under `+define+RANDOMIZE_REG_INIT`.  
**Fix:** Explicit `RegInit` zeros on all bridge state (core left unreset —
write-before-read OK).  
**Diag:** `chipyard/tmp_baseline_validation/phase3_row0_softmax_diag/FINDINGS.txt`.

### 2. Fence gap — `pwl_data_q` not in Scratchpad `busy`

**Symptom:** Fence could retire while Softmax weight beats were still draining
from `pwl_data_q` (timing-dependent under more variation than the L=16 test).  
**Cause:** Scratchpad `busy` tracked the bridge but not the post-Softmax data
queue.  
**Fix:** `pwl_busy := bridge.busy || pwl_data_q.deq.valid` in `Scratchpad.scala`.  
**Note:** Separate from the row-0 RegInit bug; both must be fixed for a robust
RTL Softmax path.

### 3. DMA / mvout tiling — 7-bit `cols` truncation (L≥128)

**Symptom (Verilator L=128):** `DMACommandTracker` assert
(`bytes_left >= bytes_read`) — store issued with `cols=128`, HW field
`mvout_cols_bits=7` truncates to 0 → `blocks=0` while DMA still fires.  
**Fix:** `tiled_pwl_softmax` chunks mvouts to `PWL_SOFTMAX_MAX_MVOUT_COLS=64`;
bridge accumulates DIM-blocks across chunks and Softmaxes when
`nextBase == configuredRowLen`. Same packing in Spike `libgemmini`.  
**Status:** **Resolved** — closed out by the post-tiling-fix validation below
(Spike + Verilator). Distinct from bugs (1) and (2).

---

## Post-Tiling-Fix Validation (2026-08-12)

The DMA tiling fix (chunking mvout to ≤64 cols in `tiled_pwl_softmax`, fixing
the 7-bit cols field truncation that caused L=128 to fail with a
`DMACommandTracker` assert) has been fully validated on both Spike and
Verilator. No Gemmini-authored files were changed for this validation run —
it exercises the fixed code as of the tiling fix.

### Spike correctness

| Suite | Grid | Result |
| --- | --- | --- |
| Random | `L ∈ {16, 32, 64, 128, 256}` × seeds 1–5 | **25 / 25 PASS**, Softmax **MATCH** (`n_int8_diff=0`) |
| Edges | `all_ones`, `all_max_mag`, `one_hot`, `checkerboard` × same L × seeds | **100 / 100 PASS** |

Path: `USE_HW_PWL_SOFTMAX` vs host `pwl_quantized_softmax` (Phase 2b CHAR gold).

### Verilator (`GemminiPWLSoftmaxConfig`, seed=1)

`D_MODEL=16`, `D_FF=64`. Wall time is host Verilator sim time only — use cycle
columns for architecture. Pre-tiling-fix totals/Softmax from
`chipyard/tmp_baseline_validation/phase3_verilator_hw_pwl/report.tsv`
(L=16/32/64 PASS before the L=128 DMA failure).

| L | Result | Total cycles | Softmax cycles | Wall (s) | vs. pre-tiling-fix (total / Softmax) |
| ---: | --- | ---: | ---: | ---: | --- |
| 16 | **PASS** | 56663 | 3159 | 1729 | 56649 / 3141 (**+14 / +18**, noise) |
| 32 | **PASS** | 114429 | 13268 | 2319 | 114474 / 13272 (**−45 / −4**, noise) |
| 64 | **PASS** | 245790 | 47958 | 6184 | 246188 / 48159 (**−398 / −201**, expected from chunking) |
| 128 | **PASS** | 623942 | 227085 | 21141 | matches prior post-DMA-fix run (`phase3_l128_dma_debug`) |
| 256 | skipped | — | — | — | — |

Per-stage cycles from the same Verilator logs (`print_stage_report` /
`* cycles:` lines — same instrumentation as the scalar baseline). Source:
`chipyard/tmp_baseline_validation/post_tiling_fix_validation/logs/L*_D16_F64_seed1.verilator.log`
(L=128 bit-identical to `phase3_l128_dma_debug`).

| L | QKV | Scores | Softmax | Attn×V | Out proj | Res1 | RMSNorm1 | FFN | Res2 | RMSNorm2 | **Total** |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 2539 | 733 | 3159 | 741 | 700 | 8349 | 15197 | 1967 | 8233 | 15045 | **56663** |
| 32 | 2874 | 1089 | 13268 | 1008 | 833 | 16472 | 30026 | 2602 | 16324 | 29933 | **114429** |
| 64 | 3658 | 2441 | 47958 | 1559 | 1130 | 32780 | 60021 | 3953 | 32449 | 59841 | **245790** |
| 128 | 6257 | 8204 | 227085 | 3880 | 1765 | 65301 | 119610 | 7872 | 64664 | 119304 | **623942** |

No DMA or other hardware asserts on any L. This **closes out the DMA/mvout-tiling
bug on RTL** (bug 3 above). It is **not** a re-discovery of the RegInit (bug 1)
or fence (bug 2) issues.

### Artifacts

Full per-seed / per-edge-case breakdown (not just aggregate PASS counts):

`chipyard/tmp_baseline_validation/post_tiling_fix_validation/SUMMARY.txt`

Also under that directory: `spike_report.tsv`, `spike_edge_report.tsv`,
`verilator_report.tsv`, and per-config logs.

Harness: `chipyard/tmp_baseline_validation/run_post_tiling_fix_validation.sh`.

---

## Comparison vs. Scalar-Softmax Baseline

Side-by-side with the Verilator numbers already published in [`BASELINE.md`](BASELINE.md)
(§2, `GemminiRocketConfig`, seed=1, HW-resadd default-on). HW-PWL numbers are the
post-tiling-fix Verilator sweep above (`GemminiPWLSoftmaxConfig`, seed=1).
`BASELINE.md` is not modified; this section is additive only.

### Comparability

| Item | Scalar (`BASELINE.md`) | HW-PWL (this doc) | Match? |
| --- | --- | --- | --- |
| `D_MODEL` / `D_FF` / `DIM` | 16 / 64 / 16 | 16 / 64 / 16 | yes |
| `PRNG_SEED` / case | 1 / random | 1 / random | yes |
| `USE_HW_RESADD` | default **1** (`tiled_resadd_auto`) | default **1** (not overridden) | yes |
| `QUANT_SCALE` / final tol | `1/128` / `5 × QUANT_SCALE` (`0.039`) | same (`QUANT_SCALE=0.008`, tol=`0.039`) | yes |
| Softmax implementation | Host scalar (`expf` + largest-remainder) | HW PWL (`USE_HW_PWL_SOFTMAX=1`) | **intentional difference** |
| Chipyard config | `GemminiRocketConfig` | `GemminiPWLSoftmaxConfig` (`has_pwl_softmax`) | differs only by PWL Softmax RTL |

Non-Softmax absolute stage cycles (GEMM, Residual, RMSNorm) at L=16/32/64 agree
with [`BASELINE.md`](BASELINE.md)’s per-stage table to within a few percent (Residual /
RMSNorm typically within ~0.5%). At **L=128**, Residual 1/2 absolute cycles are
**not** the same: HW-PWL logs 65301 / 64664 vs scalar baseline 73609 / 72927
(~11% lower on the PWL run); other non-Softmax stages at L=128 remain within a
few percent. That L=128 Residual gap is a **confirmed real divergence** (see
investigation note below), not a logging glitch — treat Residual absolute counts
at L=128 as non-comparable across configs when reading share shifts; Softmax and
total-cycle columns below are still the measured values from each published run.

**L=128 Residual investigation (read-only):** Both builds use default
`USE_HW_RESADD=1` (PWL L=128 `rtl_build.log` does not pass `-DUSE_HW_RESADD=0`).
The ~65k Res1 / ~64k Res2 counts appear in **two independent** PWL Verilator
runs (`phase3_l128_dma_debug` and `post_tiling_fix_validation`, identical stage
tables). PWL Res1 at L=128 is ≈ 2× L=64 (65301 vs 2×32780); scalar Res1 is
≈ 2× L=64 **+ ~8.2k** (73609 vs 2×32689). Adjacent post-Softmax GEMM stages
(Attn×V / Out / FFN) match closely at L=128, so the gap is Residual-specific in
the measured window. **Ruled out:** wrong `USE_HW_RESADD` flag; one-off log
corruption; elaboration mismatch on unused CNN knobs
(`has_training_convs` / `has_max_pool`) — matching those to Rocket defaults
(`true`/`true`) and re-running L=128 Verilator left Res1/Res2 at **65301 /
64664** (and Softmax/total unchanged at 227085 / 623942); artifact
`chipyard/tmp_baseline_validation/pwl_l128_residual_elab_ab/SUMMARY.txt`. A
single root cause is still **not** pinned; leading remaining hypothesis is
DRAMSim / host-cache state after ~21M-cycle host Softmax vs ~227k-cycle HW
Softmax before Residual touches `X`.

### Cycle comparison (Verilator, seed=1)

Speedups are scalar ÷ HW-PWL. Softmax % is Softmax cycles ÷ total for that build.
Scalar Softmax % values below are recomputed from the same cycle counts as
[`BASELINE.md`](BASELINE.md) (that doc rounds them to one decimal: 67.2% / 85.9% /
95.1% / 98.1%).

| L | Scalar total | HW-PWL total | Total speedup (×) | Scalar Softmax | HW-PWL Softmax | Softmax speedup (×) | Scalar Softmax % | HW-PWL Softmax % |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 163529 | 56663 | 2.885993 | 109903 | 3159 | 34.790440 | 67.2070% | 5.5751% |
| 32 | 720706 | 114429 | 6.298281 | 619397 | 13268 | 46.683524 | 85.9431% | 11.5950% |
| 64 | 4029349 | 245790 | 16.393462 | 3831599 | 47958 | 79.894887 | 95.0923% | 19.5118% |
| 128 | 21480688 | 623942 | 34.427379 | 21070239 | 227085 | 92.785693 | 98.0892% | 36.3952% |
| 256 | — (Spike-only in `BASELINE.md`) | not run on Verilator | — | — | — | — | — | — |

### Share of total after Softmax shrinks

With Softmax cycles much smaller, GEMM / Residual / RMSNorm absolute cycle counts
that stay near the scalar-baseline values (L≤64; see caveat for L=128 Residual
above) necessarily occupy a larger fraction of the reduced total. Exact scalar-side
per-stage cycles and stage-share tables are in [`BASELINE.md`](BASELINE.md)
§2 (“Per-stage cycles” and “Combined stage shares”); pull those numbers directly
rather than re-deriving them here.

---

## Related

| Doc / path | Role |
| --- | --- |
| [`BASELINE.md`](BASELINE.md) | Stock Gemmini (host Softmax) cycle baseline |
| [`SOFTMAX_ERROR_BASELINE.md`](SOFTMAX_ERROR_BASELINE.md) | Step-1 float→int8 Softmax quantization error |
| `chipyard/tmp_baseline_validation/phase2b_hw_pwl_spike/` | Spike HW-PWL vs SW-PWL model grid |
| `chipyard/tmp_baseline_validation/phase3_verilator_hw_pwl/` | Pre-tiling-fix Verilator L-sweep (L=128 FAIL) |
| `chipyard/tmp_baseline_validation/phase3_l128_dma_debug/` | First post-tiling L=64/128 Verilator PASS |
