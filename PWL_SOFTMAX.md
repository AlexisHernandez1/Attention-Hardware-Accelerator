# Hardware PWL Softmax (`GemminiPWLSoftmaxConfig`)

**Status:** Custom Q16.16 piecewise-linear Softmax elaborated into Gemmini under
`has_pwl_softmax` (separate from `GemminiRocketConfig` / I-BERT Normalizer).  
**Companion:** Host Softmax error baseline remains [`SOFTMAX_ERROR_BASELINE.md`](SOFTMAX_ERROR_BASELINE.md);
stock-Gemmini cycle baseline remains [`BASELINE.md`](BASELINE.md);
cross-build comparisons remain [`COMPARISON.md`](COMPARISON.md).  
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

For cross-build comparisons and analysis, see [`COMPARISON.md`](COMPARISON.md).

---

## Related

| Doc / path | Role |
| --- | --- |
| [`BASELINE.md`](BASELINE.md) | Stock Gemmini (host Softmax) cycle baseline |
| [`COMPARISON.md`](COMPARISON.md) | Cross-build Softmax cycle / accuracy / residual analysis |
| [`SOFTMAX_ERROR_BASELINE.md`](SOFTMAX_ERROR_BASELINE.md) | Step-1 float→int8 Softmax quantization error |
| `chipyard/tmp_baseline_validation/phase2b_hw_pwl_spike/` | Spike HW-PWL vs SW-PWL model grid |
| `chipyard/tmp_baseline_validation/phase3_verilator_hw_pwl/` | Pre-tiling-fix Verilator L-sweep (L=128 FAIL) |
| `chipyard/tmp_baseline_validation/phase3_l128_dma_debug/` | First post-tiling L=64/128 Verilator PASS |
