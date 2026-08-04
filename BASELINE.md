# Official Performance Baseline

**Status:** Authoritative baseline for this project going forward.  
**Build:** Current permanent `transformer_block_test` — GEMM stages use counters-inline timing with deferred utilization printf. No `GEMM_HW_UTIL_DEBUG` / `GEMM_COUNTER_OVERHEAD_TEST` paths.  
**Config:** Single-head decoder block, `D_MODEL=16`, `D_FF=64`, `DIM=16`, calibrated Q/K scales / `SCORE_DEQUANT` / `RMSNORM_GAIN` defaults. Softmax int8 mass via largest-remainder. Final check tolerance = `5 × QUANT_SCALE`.  
**Sources:** Spike grid logs under `chipyard/tmp_baseline_validation/phase1_spike/`; Verilator under `chipyard/tmp_baseline_validation/phase2_verilator/`. Companion tables also in [`sweeps/l_sweep/README.md`](sweeps/l_sweep/README.md).

---

## Architectural / build conditions (read first)

This baseline measures **current Gemmini hardware usage, unmodified** — not an upper bound on what a specialized attention accelerator can achieve.

| Stage | Where it runs today | Notes |
| --- | --- | --- |
| **QKV, Scores, Attn×V, Output proj, FFN** | **Real Gemmini hardware** via `tiled_matmul_auto` (WS) | Confirmed accelerator path — not a CPU matmul fallback. |
| **Softmax** | **Scalar C on the host core** | Gemmini’s Normalizer / I-BERT path (`Activation` includes `SOFTMAX`) exists upstream but is **off** here (`has_normalizations=false`) and is documented upstream as experimental. Even if enabled, its integer Normalizer path does **not** match this project’s float + largest-remainder quantized softmax. **No hardware softmax is in use.** |
| **RMSNorm** | **Scalar C on the host core** | Gemmini has **no RMSNorm hardware mode** — only LayerNorm (also disabled). There is no hardware RMSNorm to turn on. |
| **Residual 1 / 2** | **Scalar C on the host core** | Gemmini has a working hardware residual-add path (`tiled_resadd_auto`) that this baseline **does not use**. No technical blocker is documented — a near-term optimization, not yet applied. |

**Consequence (expected, not mysterious):** Softmax, RMSNorm, and Residual dominate total cycles because those three stages run on the host core rather than dedicated hardware. Upcoming project work — a hardware softmax unit, a hardware RMSNorm unit, and moving residual adds to `tiled_resadd_auto` — targets exactly these three now-quantified bottlenecks.

GEMM’s small share of total time means the mesh is fast relative to host Softmax/RMSNorm/Residual — not that Gemmini is idle or unused.

**Out of scope for this baseline:** the archived pre-counters-inline L=256 Verilator log under [`sweeps/l_sweep/legacy_pre_counters/`](sweeps/l_sweep/legacy_pre_counters/) (different GEMM timing path; not comparable; not merged into any table below).

---

## Spike results — full correctness + timing grid

**Simulator:** Spike + Gemmini functional model.  
**Grid:** 8 cases × 5 lengths × 5 seeds = **200 configs**.  
**Result:** **200 / 200 PASS.**

Cases: `random` (float-gold, tolerance `5×QUANT_SCALE`) and edges `all_zeros`, `all_ones`, `all_max_mag`, `one_hot`, `checkerboard`, `all_negative`, `near_zero` (case-specific assertions).  
`L ∈ {16, 32, 64, 128, 256}`, `PRNG_SEED ∈ {1…5}`, `D_MODEL=16`, `D_FF=64`.

Spike cycle counts are useful for relative stage share and scaling, but are **not** RTL-accurate. Spike PE utilization counters are **not** cycle-accurate and are **omitted** from this baseline (do not read Spike `exe_act` / `mesh_util` as hardware utilization).

### Correctness summary

| Case | PASS | Saturation banner (int8 range hit clip) | Notes |
| --- | ---: | --- | --- |
| random | 25/25 | **none** | Gold-tolerant; no sat on GEMM/softmax or Residual/RMSNorm software rails |
| all_zeros | 25/25 | none | |
| all_ones | 25/25 | 25/25 (intentional; Q/K and/or Residual 1) | Edge assertions PASS; sat fail skipped / clamp path for rail cases |
| all_max_mag | 25/25 | 25/25 (intentional) | Same |
| one_hot | 25/25 | 25/25 (scores / softmax / Residual 1; sometimes Final RMSNorm) | Expected collapse; assertions PASS |
| checkerboard | 25/25 | 25/25 (Q/K and/or Residual 1) | Preclip bound enforced in-pipeline; PASS |
| all_negative | 25/25 | **none** | See margin note below |
| near_zero | 25/25 | none | |
| **Total** | **200/200** | | |

**Historical all_negative / L=256 Residual-1 preclip margin:** Earlier audits reported Residual-1 preclip peaks above the ~115 band for some L=256 seeds. **Under this baseline grid that failure does not reproduce:** all five L=256 `all_negative` seeds **PASS**, with Residual-1 `max_abs` = 95 / 98 / 96 / 84 / 99 (all ≤ 115) and `preclip_fail=no`. Treat the old margin exceedance as a prior, data-dependent issue that is **not present** in these validated results — not as an open FAIL on this baseline.

Clip conventions used in logs: hardware GEMM / softmax banners use true int8 rails (−128 / 127); software Residual 1/2, RMSNorm 1, Final RMSNorm use quantize clip ±127 (−127 / 127).

### Random baseline — per-stage cycles (seed=1)

Authoritative seed=1 stage breakdown (same seed as Verilator). Percents are of that row’s total.

| L | QKV | Scores | Softmax | Attn×V | Out proj | Res1 | RMSNorm1 | FFN | Res2 | RMSNorm2 | **Total** | Result |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 16 | 1071 (1.2%) | 361 (0.4%) | 62192 (66.9%) | 360 (0.4%) | 359 (0.4%) | 6568 (7.1%) | 7402 (8.0%) | 718 (0.8%) | 6570 (7.1%) | 7403 (8.0%) | **93004** | PASS |
| 32 | 1090 (0.3%) | 368 (0.1%) | 353258 (85.8%) | 366 (0.1%) | 365 (0.1%) | 13095 (3.2%) | 14713 (3.6%) | 730 (0.2%) | 13095 (3.2%) | 14712 (3.6%) | **411792** | PASS |
| 64 | 1093 (0.05%) | 369 (0.02%) | 2208986 (95.1%) | 367 (0.02%) | 366 (0.02%) | 26146 (1.1%) | 29332 (1.3%) | 732 (0.03%) | 26151 (1.1%) | 29336 (1.3%) | **2322878** | PASS |
| 128 | 1627 (0.01%) | 1003 (0.01%) | 13462849 (98.3%) | 544 (0.004%) | 543 (0.004%) | 54323 (0.4%) | 58599 (0.4%) | 1081 (0.01%) | 54331 (0.4%) | 58607 (0.4%) | **13693507** | PASS |
| 256 | 2251 (0.002%) | 2213 (0.002%) | 90329500 (99.5%) | 752 (0.001%) | 751 (0.001%) | 104511 (0.1%) | 117106 (0.1%) | 1491 (0.002%) | 104519 (0.1%) | 117114 (0.1%) | **90780208** | PASS |

### Random baseline — total cycles across seeds 1–5

All 25 random configs PASS. Softmax (and thus total) varies modestly with seed via attention sharpness.

| L | seed1 | seed2 | seed3 | seed4 | seed5 | min | max | mean |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 93004 | 93879 | 93176 | 92769 | 94442 | 92769 | 94442 | 93454 |
| 32 | 411792 | 403676 | 405046 | 407464 | 412737 | 403676 | 412737 | 408143 |
| 64 | 2322878 | 2291205 | 2185647 | 2219797 | 2272303 | 2185647 | 2322878 | 2258366 |
| 128 | 13693507 | 13218969 | 12569338 | 12787765 | 13107375 | 12569338 | 13693507 | 13075391 |
| 256 | 90780208 | 87289845 | 81481936 | 84356552 | 85004673 | 81481936 | 90780208 | 85782643 |

### Softmax distribution (random, seed=1)

| L | mean_maxp | mean_entropy | ln(L) | uniform 1/L |
| ---: | ---: | ---: | ---: | ---: |
| 16 | 0.208 | 2.430 | 2.773 | 0.0625 |
| 32 | 0.129 | 3.133 | 3.466 | 0.03125 |
| 64 | 0.080 | 3.814 | 4.159 | 0.015625 |
| 128 | 0.051 | 4.492 | 4.852 | 0.0078125 |
| 256 | 0.031 | 5.174 | 5.545 | 0.00390625 |

Entropy stays below ln(L); mean_maxp stays above uniform 1/L — not a flat “broken softmax” regime on this baseline.

### Logical traffic estimate (random; same for Spike seed=1 and Verilator seed=1)

Estimated operand/output bytes (not DMA counters):

| L | Total estimated bytes |
| ---: | ---: |
| 16 | 7936 |
| 32 | 14848 |
| 64 | 34816 |
| 128 | 99328 |
| 256 | 326656 |

Scores + attention-weight traffic dominate growth (~L²).

---

## Verilator results — cycle-accurate RTL (`GemminiRocketConfig`)

**Config:** random, `PRNG_SEED=1`, `L ∈ {16, 32, 64, 128}`, `SKIP_GOLD=1` + Spike-exported `expected_final`.  
**Result:** **4 / 4 PASS.** No saturation. No `+max-cycles` timeout.  
**L=256:** not part of this baseline (incomplete counters-inline attempt purged; legacy archive not merged).

Utilization below is from Gemmini perf counters with **batched** summary (no printf inside timed GEMM windows). These numbers are meaningful on RTL.

### Per-stage cycles (seed=1)

| L | QKV | Scores | Softmax | Attn×V | Out proj | Res1 | RMSNorm1 | FFN | Res2 | RMSNorm2 | **Total** | Result |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 16 | 2541 (1.5%) | 747 (0.4%) | 109919 (65.0%) | 725 (0.4%) | 735 (0.4%) | 11147 (6.6%) | 15192 (9.0%) | 2072 (1.2%) | 11001 (6.5%) | 15043 (8.9%) | **169122** | PASS |
| 32 | 2839 (0.4%) | 1109 (0.2%) | 619399 (84.7%) | 1045 (0.1%) | 832 (0.1%) | 21963 (3.0%) | 30084 (4.1%) | 2659 (0.4%) | 21813 (3.0%) | 29901 (4.1%) | **731644** | PASS |
| 64 | 3646 (0.1%) | 2462 (0.1%) | 3831625 (94.6%) | 1889 (0.05%) | 1084 (0.03%) | 43893 (1.1%) | 59205 (1.5%) | 3902 (0.1%) | 43766 (1.1%) | 59007 (1.5%) | **4050479** | PASS |
| 128 | 6010 (0.03%) | 8295 (0.04%) | 21067285 (98.0%) | 3751 (0.02%) | 1787 (0.01%) | 87505 (0.4%) | 118007 (0.5%) | 7789 (0.04%) | 86790 (0.4%) | 117943 (0.5%) | **21505162** | PASS |

Softmax mean_maxp / mean_entropy match Spike seed=1 at each L (same logits path). Saturation: **none** on all four runs.

### GEMM utilization (Verilator only)

| L | Stage | wall | exe_act | ideal | exe_busy | mesh_util |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 16 | QKV | 2541 | 90 | 48 | 3% | 53% |
| 16 | Scores | 747 | 45 | 16 | 6% | 35% |
| 16 | Attn×V | 725 | 30 | 16 | 4% | 53% |
| 16 | Out | 735 | 30 | 16 | 4% | 53% |
| 16 | FFN | 2072 | 198 | 128 | 9% | 64% |
| 32 | QKV | 2839 | 158 | 96 | 5% | 60% |
| 32 | Scores | 1109 | 105 | 64 | 9% | 60% |
| 32 | Attn×V | 1045 | 104 | 64 | 9% | 61% |
| 32 | Out | 832 | 48 | 32 | 5% | 66% |
| 32 | FFN | 2659 | 383 | 256 | 14% | 66% |
| 64 | QKV | 3646 | 340 | 192 | 9% | 56% |
| 64 | Scores | 2462 | 361 | 256 | 14% | 70% |
| 64 | Attn×V | 1889 | 341 | 256 | 18% | 75% |
| 64 | Out | 1084 | 111 | 64 | 10% | 57% |
| 64 | FFN | 3902 | 733 | 512 | 18% | 69% |
| 128 | QKV | 6010 | 696 | 384 | 11% | 55% |
| 128 | Scores | 8295 | 1350 | 1024 | 16% | 75% |
| 128 | Attn×V | 3751 | 1951 | 1024 | 52% | 52% |
| 128 | Out | 1787 | 234 | 128 | 13% | 54% |
| 128 | FFN | 7789 | 1524 | 1024 | 19% | 67% |

Mesh util is typically ~50–75% while execute-busy stays low — GEMM walls are short relative to Softmax, so utilization is about mesh efficiency during those short bursts, not about reclaiming Softmax’s share of the block.

### Wall-clock simulation time (operational only — **not** hardware performance)

| L | Wall-clock (s) | Wall-clock (h) | vs L=16 |
| ---: | ---: | ---: | ---: |
| 16 | 1213 | 0.34 | 1.0× |
| 32 | 2477 | 0.69 | 2.0× |
| 64 | 7403 | 2.06 | 6.1× |
| 128 | 28295 | 7.86 | 23.3× |

Host Verilator speed ≠ DUT cycle efficiency. Use cycle tables above for architecture; use this table only for planning sim budgets.

---

## Findings (grounded in the numbers above)

### 1. Softmax (host scalar) is the bottleneck at every L, and its share grows with L

On Verilator seed=1: Softmax is **65% → 85% → 95% → 98%** of total as L goes 16 → 32 → 64 → 128. Spike seed=1 matches the pattern through L=256 (**67% → 86% → 95% → 98% → 99.5%**).

When L doubles, Softmax cycles grow by ~5.5–6.7× (not a clean 4×), and totals follow Softmax. That is the direct cost of L×L `expf` / largest-remainder work on the host — exactly the Softmax row in the conditions table.

### 2. Residual + RMSNorm are the next host-scalar costs; they grow ~linearly with L

At L=16 (Verilator), Res1+RMS1+Res2+RMS2 ≈ **31%** of total. By L=128 that combined host-norm/residual block is only ~1.8% because Softmax has absorbed almost everything — but absolute Residual/RMSNorm cycles still rise roughly with L (e.g. Res1 11 147 → 87 505 from L=16→128). They remain the obvious next targets after Softmax for hardware offload / `tiled_resadd_auto`.

### 3. GEMM’s share is small and shrinks with L — hardware is fast, not missing

Combined GEMM stages (QKV+Scores+Attn×V+Out+FFN):

| L | Spike seed=1 GEMM share | Verilator seed=1 GEMM share |
| ---: | ---: | ---: |
| 16 | 3.1% | **4.0%** |
| 32 | 0.7% | 1.2% |
| 64 | 0.1% | 0.3% |
| 128 | ~0.03% | 0.1% |
| 256 | ~0.008% | — |

The oft-quoted “~3–4%” applies near **L=16**. At larger L the GEMM share falls well below 1%. Absolute GEMM cycles still grow with L (more tiles), but Softmax grows much faster. This is Gemmini doing its matmuls quickly relative to host Softmax — underutilized as a **fraction of block time**, not unused.

### 4. Correctness: clean grid; prior all_negative margin not present here

200/200 Spike PASS; 4/4 Verilator PASS; random baseline has **zero** saturation banners. Edge-case saturation banners appear only on intentional rail / one-hot / checkerboard fills and still PASS under case assertions. The previously discussed all_negative L=256 Residual-1 preclip margin exceedance **does not appear** in this grid (all five seeds in-band and PASS).

### 5. Verilator wall-clock (sim speed) grows slower than cycle count

Cycles grow ~4.3× / 5.5× / 5.3× per L doubling; wall-clock grows ~2.0× / 6.1× / 23× from L=16→32→64→128 cumulative factors. Use wall times only for ops planning (~8 h for L=128; L=256 under this build still outstanding).

---

## Plain-language summary

This baseline is **unmodified Gemmini usage**: real hardware for GEMMs only; Softmax, RMSNorm, and Residual on the host. That is why Softmax alone is already two-thirds of RTL cycles at L=16 and essentially the whole block by L=128–256. GEMM stays a few percent at small L and then vanishes as a share — the accelerator is working; the host scalar stages are the clock. Correctness is clean on the full Spike grid and on Verilator L=16–128 seed=1. The next engineering moves (hardware softmax, hardware RMSNorm, hardware residual-add) map one-to-one onto the three stages that own the cycle budget.
