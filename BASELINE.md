# Official Performance Baseline

**Status:** Authoritative baseline for this project going forward.  
**Build:** Current permanent `transformer_block_test` — GEMM stages use counters-inline timing with deferred utilization printf; **Residual 1/2 use Gemmini `tiled_resadd_auto` by default** (`USE_HW_RESADD=1`, identity mvin/acc scales, `relu=false`, WS, with a `-128→-127` post-pass to match the former scalar clip). Override with `-DUSE_HW_RESADD=0` only for A/B against the old scalar residual path.  
**Config:** Single-head decoder block, `D_MODEL=16`, `D_FF=64`, `DIM=16`, calibrated Q/K scales / `SCORE_DEQUANT` / `RMSNORM_GAIN` defaults. Softmax int8 mass via largest-remainder. Final check tolerance = `5 × QUANT_SCALE`.  
**Sources:** Spike HW-resadd validation under `chipyard/tmp_baseline_validation/hw_resadd_default_spike/` (and prior on/off compare under `hw_resadd_spike/`); Verilator under `chipyard/tmp_baseline_validation/phase2_verilator_hw_resadd/`. Companion tables also in [`sweeps/l_sweep/README.md`](sweeps/l_sweep/README.md).

**Verilator is the sole authoritative source for cycle counts, stage share, and PE utilization.** Spike is used for correctness / regression only.

---

## Architectural / build conditions (read first)

This baseline measures **current Gemmini hardware usage, unmodified** — not an upper bound on what a specialized attention accelerator can achieve.

| Stage | Where it runs today | Notes |
| --- | --- | --- |
| **QKV, Scores, Attn×V, Output proj, FFN** | **Real Gemmini hardware** via `tiled_matmul_auto` (WS) | Confirmed accelerator path — not a CPU matmul fallback. |
| **Residual 1 / 2** | **Real Gemmini hardware** via `tiled_resadd_auto` (WS) | Default since `USE_HW_RESADD=1`. Same `QUANT_SCALE` int8 operands → identity scales; post-pass maps HW negative sat (−128) to scalar clip (−127) for bit-identical tensors. |
| **Softmax** | **Scalar C on the host core** | Gemmini’s Normalizer / I-BERT path (`Activation` includes `SOFTMAX`) exists upstream but is **off** here (`has_normalizations=false`) and is documented upstream as experimental. Even if enabled, its integer Normalizer path does **not** match this project’s float + largest-remainder quantized softmax. **No hardware softmax is in use.** |
| **RMSNorm** | **Scalar C on the host core** | Gemmini has **no RMSNorm hardware mode** — only LayerNorm (also disabled). There is no hardware RMSNorm to turn on. |

**Consequence:** Softmax (host) still dominates RTL cycles at every L. Residual is no longer a host-scalar stage; RMSNorm remains the next host-scalar target after Softmax. GEMM’s share of total time stays small — the mesh is fast relative to Softmax, not idle.

**Out of scope for this baseline:** the archived pre-counters-inline L=256 Verilator log under [`sweeps/l_sweep/legacy_pre_counters/`](sweeps/l_sweep/legacy_pre_counters/) (different GEMM timing path; not comparable; not merged into any table below). **L=256 Verilator** was never completed under the counters-inline / HW-resadd builds and is **Spike-only**.

**Not measured in these runs:** DMA / memory-stall cycle counters (no such fields appear in the Spike or Verilator test logs). Utilization below is only what `print_gemm_util_summary` records for GEMM stages (`exe_act`, `loop_act`, `ideal`, `exe_busy`, `mesh_util`). Residual-add does not emit a separate util row.

---

## 1. Spike validation (correctness / regression only)

**Simulator:** Spike + Gemmini functional model.  
**Role:** Correctness and HW-vs-scalar residual regression — **not** performance. Spike `rdcycle` stage times and Spike PE util counters are **functional / not cycle-accurate**; they are **not** used as baseline performance numbers (Spike util rows in logs are omitted here entirely).

### HW-resadd random grid (current default)

**Grid:** random (`EDGE_CASE_ID=0`) × `L ∈ {16, 32, 64, 128, 256}` × `PRNG_SEED ∈ {1…5}` = **25 configs**.  
**Build:** default `USE_HW_RESADD=1` (no flag override).  
**Result:** **25 / 25 PASS.** Zero saturation banners. Gold check tolerance `5 × QUANT_SCALE`.

Prior A/B on the same 25 configs with explicit `-DUSE_HW_RESADD=0` vs `=1`: **25/25 PASS on both paths**, **0 disagreements** on Residual 1 / Residual 2 / Final RMSNorm int8 ranges (`max_abs` and printed `[lo,hi]` match every config after the `-128→-127` post-pass).

| L | seed | Result | Sat banner | Res1 max_abs | Res2 max_abs | Final RMSNorm max_abs | mean_maxp | mean_entropy |
| ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 16 | 1 | PASS | none | 67 | 82 | 81 | 0.208 | 2.430 |
| 16 | 2 | PASS | none | 65 | 84 | 84 | 0.224 | 2.432 |
| 16 | 3 | PASS | none | 65 | 90 | 90 | 0.356 | 2.103 |
| 16 | 4 | PASS | none | 63 | 88 | 87 | 0.241 | 2.343 |
| 16 | 5 | PASS | none | 64 | 84 | 84 | 0.193 | 2.488 |
| 32 | 1 | PASS | none | 66 | 83 | 84 | 0.129 | 3.133 |
| 32 | 2 | PASS | none | 65 | 84 | 84 | 0.151 | 3.046 |
| 32 | 3 | PASS | none | 65 | 91 | 90 | 0.193 | 2.902 |
| 32 | 4 | PASS | none | 65 | 87 | 86 | 0.150 | 3.069 |
| 32 | 5 | PASS | none | 64 | 87 | 86 | 0.136 | 3.111 |
| 64 | 1 | PASS | none | 66 | 91 | 90 | 0.080 | 3.814 |
| 64 | 2 | PASS | none | 65 | 91 | 90 | 0.099 | 3.736 |
| 64 | 3 | PASS | none | 65 | 93 | 93 | 0.117 | 3.602 |
| 64 | 4 | PASS | none | 65 | 93 | 94 | 0.109 | 3.673 |
| 64 | 5 | PASS | none | 65 | 87 | 86 | 0.088 | 3.765 |
| 128 | 1 | PASS | none | 65 | 95 | 95 | 0.051 | 4.492 |
| 128 | 2 | PASS | none | 66 | 103 | 103 | 0.060 | 4.427 |
| 128 | 3 | PASS | none | 65 | 109 | 109 | 0.070 | 4.306 |
| 128 | 4 | PASS | none | 65 | 93 | 94 | 0.063 | 4.375 |
| 128 | 5 | PASS | none | 65 | 92 | 92 | 0.058 | 4.408 |
| 256 | 1 | PASS | none | 65 | 101 | 102 | 0.031 | 5.174 |
| 256 | 2 | PASS | none | 66 | 103 | 103 | 0.035 | 5.112 |
| 256 | 3 | PASS | none | 66 | 110 | 110 | 0.047 | 4.982 |
| 256 | 4 | PASS | none | 65 | 112 | 111 | 0.041 | 5.039 |
| 256 | 5 | PASS | none | 66 | 93 | 93 | 0.039 | 5.062 |

Softmax distribution (seed=1): entropy stays below `ln(L)`; `mean_maxp` stays above uniform `1/L` — not a flat “broken softmax” regime. Values match Verilator seed=1 at L=16/32/64/128 (same logits path).

**Spike-only sanity check (non-authoritative timing):** On the prior A/B logs for L=16 seed=1, Residual add 1 was **6568** Spike cycles with scalar residual (`USE_HW_RESADD=0`) vs **5198** with HW residual (`USE_HW_RESADD=1`). Treat this only as a functional-sim illustration that the HW path is exercised inside the stage timer — **not** as a performance result.

**Earlier 200-config edge grid** (8 cases × 5 L × 5 seeds, scalar residual era) is historical correctness context under `phase1_spike/`; it was not re-run for this HW-resadd default promotion. Random HW-resadd tensors match scalar residual on the 25-config A/B above.

Clip conventions in logs: hardware GEMM / softmax banners use true int8 rails (−128 / 127); Residual 1/2 and RMSNorm use software clip ±127 after the residual post-pass (−127 / 127).

### Logical traffic estimate (correctness-adjacent; same seed=1 logs)

Estimated operand/output bytes printed by the test (not DMA counters). Verilator seed=1 matches Spike seed=1 at each L in `{16,32,64,128}`:

| L | Total estimated bytes |
| ---: | ---: |
| 16 | 7936 |
| 32 | 14848 |
| 64 | 34816 |
| 128 | 99328 |
| 256 | 326656 (Spike seed=1 only; no Verilator L=256) |

Scores + attention-weight traffic dominate growth (~L²).

---

## 2. Verilator baseline (authoritative performance — `GemminiRocketConfig`)

**Config:** random, `PRNG_SEED=1`, `L ∈ {16, 32, 64, 128}`, default `USE_HW_RESADD=1`, `SKIP_GOLD=1` + Spike-exported `expected_final` from the same default build.  
**Result:** **4 / 4 PASS.** No saturation banners. No `+max-cycles` timeout.  
**L=256:** not run on Verilator (Spike-only); do not compare L=256 here.

### Headline cycles + wall-clock (seed=1)

| L | Result | Total cycles | Residual 1 | Residual 2 | Wall (s) | Wall (h) |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 16 | PASS | 163529 | 8387 | 8265 | 1112 | 0.31 |
| 32 | PASS | 720706 | 16489 | 16297 | 2298 | 0.64 |
| 64 | PASS | 4029349 | 32689 | 32391 | 7861 | 2.18 |
| 128 | PASS | 21480688 | 73609 | 72927 | 28778 | 7.99 |

Wall-clock is host Verilator sim time only — **not** DUT performance. Use cycle columns for architecture.

### Per-stage cycles (seed=1)

Percents are of that row’s total.

| L | QKV | Scores | Softmax | Attn×V | Out proj | Res1 | RMSNorm1 | FFN | Res2 | RMSNorm2 | **Total** | Result |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 16 | 2565 (1.6%) | 746 (0.5%) | 109903 (**67.2%**) | 724 (0.4%) | 713 (0.4%) | 8387 (5.1%) | 15196 (9.3%) | 1990 (1.2%) | 8265 (5.1%) | 15040 (9.2%) | **163529** | PASS |
| 32 | 2875 (0.4%) | 1075 (0.1%) | 619397 (**85.9%**) | 1061 (0.1%) | 818 (0.1%) | 16489 (2.3%) | 30068 (4.2%) | 2690 (0.4%) | 16297 (2.3%) | 29936 (4.2%) | **720706** | PASS |
| 64 | 3700 (0.1%) | 2446 (0.1%) | 3831599 (**95.1%**) | 1920 (0.05%) | 1045 (0.03%) | 32689 (0.8%) | 59899 (1.5%) | 3937 (0.1%) | 32391 (0.8%) | 59723 (1.5%) | **4029349** | PASS |
| 128 | 6156 (0.03%) | 8260 (0.04%) | 21070239 (**98.1%**) | 3807 (0.02%) | 1784 (0.01%) | 73609 (0.3%) | 118069 (0.5%) | 7888 (0.04%) | 72927 (0.3%) | 117949 (0.5%) | **21480688** | PASS |

Softmax mean_maxp / mean_entropy match Spike seed=1 at each L. Combined stage shares (seed=1):

| L | Softmax | RMSNorm 1+2 | Residual 1+2 | All GEMM (QKV+Scores+Attn×V+Out+FFN) |
| ---: | ---: | ---: | ---: | ---: |
| 16 | 67.2% | 18.5% | 10.2% | 4.1% |
| 32 | 85.9% | 8.3% | 4.5% | 1.2% |
| 64 | 95.1% | 3.0% | 1.6% | 0.3% |
| 128 | 98.1% | 1.1% | 0.7% | 0.1% |

For cross-build comparisons and analysis (including vs prior scalar-residual Verilator, Softmax HW paths, and narrative conclusions), see [`COMPARISON.md`](COMPARISON.md).

### GEMM utilization (Verilator only; seed=1)

From batched Gemmini perf counters (`EXE_ACTIVE_CYCLE`, `LOOP_MATMUL_ACTIVE_CYCLES`; no printf inside timed GEMM windows). Residual-add has **no** util row in these logs.

| L | Stage | wall | exe_act | loop_act | ideal | exe_busy | mesh_util |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | QKV | 2565 | 90 | 66 | 48 | 3% | 53% |
| 16 | Scores | 746 | 45 | 22 | 16 | 6% | 35% |
| 16 | Attn×V | 724 | 30 | 22 | 16 | 4% | 53% |
| 16 | Out | 713 | 30 | 22 | 16 | 4% | 53% |
| 16 | FFN | 1990 | 202 | 62 | 128 | 10% | 63% |
| 32 | QKV | 2875 | 160 | 81 | 96 | 5% | 60% |
| 32 | Scores | 1075 | 105 | 34 | 64 | 9% | 60% |
| 32 | Attn×V | 1061 | 104 | 32 | 64 | 9% | 61% |
| 32 | Out | 818 | 52 | 27 | 32 | 6% | 61% |
| 32 | FFN | 2690 | 362 | 288 | 256 | 13% | 70% |
| 64 | QKV | 3700 | 347 | 324 | 192 | 9% | 55% |
| 64 | Scores | 2446 | 361 | 1372 | 256 | 14% | 70% |
| 64 | Attn×V | 1920 | 341 | 1092 | 256 | 17% | 75% |
| 64 | Out | 1045 | 111 | 105 | 64 | 10% | 57% |
| 64 | FFN | 3937 | 740 | 2160 | 512 | 18% | 69% |
| 128 | QKV | 6156 | 688 | 2871 | 384 | 11% | 55% |
| 128 | Scores | 8260 | 1342 | 7379 | 1024 | 16% | 76% |
| 128 | Attn×V | 3807 | 1955 | 2875 | 1024 | 51% | 52% |
| 128 | Out | 1784 | 234 | 916 | 128 | 13% | 54% |
| 128 | FFN | 7888 | 1539 | 5756 | 1024 | 19% | 66% |

Mesh util is typically ~50–75% while execute-busy stays low — GEMM walls remain short relative to Softmax.

### Seed=1 int8 ranges (Verilator; correctness cross-check)

No saturation banners. Residual / Final max_abs stay inside the validated preclip band.

| L | Res1 max_abs | Res2 max_abs | Final RMSNorm max_abs | Scores max_abs | Softmax wts max_abs |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 67 | 82 | 81 | 22 | 56 |
| 32 | 66 | 83 | 84 | 24 | 39 |
| 64 | 66 | 91 | 90 | 24 | 23 |
| 128 | 65 | 95 | 95 | 26 | 21 |
