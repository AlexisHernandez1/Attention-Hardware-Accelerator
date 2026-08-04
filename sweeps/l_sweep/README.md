# Official Sequence-Length Sweep Baseline (seed=1 focus + full Spike grid)

**Authoritative writeup:** [`../../BASELINE.md`](../../BASELINE.md) (full Spike 200-config grid, Verilator L=16–128, analysis).  
This file keeps the L-sweep tables next to the sweep tooling.

---

## Architectural / build conditions (read first)

This baseline is **current Gemmini hardware usage, unmodified** — not a claim about best possible accelerator performance.

| Stage | Where it runs | Implication |
| --- | --- | --- |
| QKV / Scores / Attn×V / Output proj / FFN | **Gemmini hardware** (`tiled_matmul_auto`) | Real accelerator path. |
| Softmax | **Host scalar C** | Upstream Normalizer/`SOFTMAX` exists but is **disabled** (`has_normalizations=false`) and experimental; integer path ≠ this project’s float + largest-remainder softmax. **No HW softmax in use.** |
| RMSNorm | **Host scalar C** | Gemmini has **no RMSNorm mode** (only disabled LayerNorm). **No HW RMSNorm available.** |
| Residual 1 / 2 | **Host scalar C** | `tiled_resadd_auto` exists and works but is **unused** here — near-term optimization not yet applied. |

Softmax / RMSNorm / Residual dominating cycles is the expected result of those three stages running on the host. Project work (HW softmax, HW RMSNorm, residual → `tiled_resadd_auto`) targets those bottlenecks.

**Build:** permanent counters-inline GEMM timing (deferred util printf). No `GEMM_HW_UTIL_DEBUG` / overhead-test flags.  
**Fixed dims:** `D_MODEL=16`, `D_FF=64`. Varied `SEQ_LEN`. Independent per-tensor PRNG from `PRNG_SEED`.  
**Do not merge** [`legacy_pre_counters/`](legacy_pre_counters/) L=256 Verilator artifacts into these tables (different GEMM timing path; see that README).

---

## Spike results

Spike + Gemmini functional model. Cycle shares are informative; Spike PE util counters are **not** cycle-accurate and are omitted.

### Full grid correctness (8 cases × L ∈ {16…256} × seeds 1–5)

**200 / 200 PASS.**

| Case | PASS | Saturation banner |
| --- | ---: | --- |
| random | 25/25 | none |
| all_zeros | 25/25 | none |
| all_ones | 25/25 | 25/25 (intentional rails) |
| all_max_mag | 25/25 | 25/25 (intentional rails) |
| one_hot | 25/25 | 25/25 (expected) |
| checkerboard | 25/25 | 25/25 (expected) |
| all_negative | 25/25 | none |
| near_zero | 25/25 | none |

Prior all_negative L=256 Residual-1 preclip margin exceedances **do not appear** in this grid (L=256 seeds 1–5 all PASS; Residual-1 `max_abs` 84–99). See `BASELINE.md`.

### Random seed=1 — per-stage cycles

| L | QKV | Scores | Softmax | Attn×V | Out | Res1 | RMS1 | FFN | Res2 | RMS2 | Total | Result |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 16 | 1071 (1%) | 361 (0%) | 62192 (67%) | 360 (0%) | 359 (0%) | 6568 (7%) | 7402 (8%) | 718 (1%) | 6570 (7%) | 7403 (8%) | 93004 | PASS |
| 32 | 1090 (0%) | 368 (0%) | 353258 (86%) | 366 (0%) | 365 (0%) | 13095 (3%) | 14713 (4%) | 730 (0%) | 13095 (3%) | 14712 (4%) | 411792 | PASS |
| 64 | 1093 (0%) | 369 (0%) | 2208986 (95%) | 367 (0%) | 366 (0%) | 26146 (1%) | 29332 (1%) | 732 (0%) | 26151 (1%) | 29336 (1%) | 2322878 | PASS |
| 128 | 1627 (0%) | 1003 (0%) | 13462849 (98%) | 544 (0%) | 543 (0%) | 54323 (0%) | 58599 (0%) | 1081 (0%) | 54331 (0%) | 58607 (0%) | 13693507 | PASS |
| 256 | 2251 (0%) | 2213 (0%) | 90329500 (99%) | 752 (0%) | 751 (0%) | 104511 (0%) | 117106 (0%) | 1491 (0%) | 104519 (0%) | 117114 (0%) | 90780208 | PASS |

### Random — total cycles, all seeds (all PASS)

| L | seed1 | seed2 | seed3 | seed4 | seed5 | mean |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 93004 | 93879 | 93176 | 92769 | 94442 | 93454 |
| 32 | 411792 | 403676 | 405046 | 407464 | 412737 | 408143 |
| 64 | 2322878 | 2291205 | 2185647 | 2219797 | 2272303 | 2258366 |
| 128 | 13693507 | 13218969 | 12569338 | 12787765 | 13107375 | 13075391 |
| 256 | 90780208 | 87289845 | 81481936 | 84356552 | 85004673 | 85782643 |

### Softmax stats (random seed=1)

| L | mean_maxp | mean_entropy | ln(L) |
| ---: | ---: | ---: | ---: |
| 16 | 0.208 | 2.430 | 2.773 |
| 32 | 0.129 | 3.133 | 3.466 |
| 64 | 0.080 | 3.814 | 4.159 |
| 128 | 0.051 | 4.492 | 4.852 |
| 256 | 0.031 | 5.174 | 5.545 |

### Traffic (logical estimate, bytes)

| L | Total |
| ---: | ---: |
| 16 | 7936 |
| 32 | 14848 |
| 64 | 34816 |
| 128 | 99328 |
| 256 | 326656 |

---

## Verilator results (`GemminiRocketConfig`)

Random seed=1, L=16/32/64/128, Spike-exported expected int8. Counters-inline GEMM; util batched after timers.

**4 / 4 PASS.** No sat. No timeout. L=256 not in this baseline.

### Per-stage cycles

| L | QKV | Scores | Softmax | Attn×V | Out | Res1 | RMS1 | FFN | Res2 | RMS2 | Total | Result |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 16 | 2541 (2%) | 747 (0%) | 109919 (65%) | 725 (0%) | 735 (0%) | 11147 (7%) | 15192 (9%) | 2072 (1%) | 11001 (7%) | 15043 (9%) | 169122 | PASS |
| 32 | 2839 (0%) | 1109 (0%) | 619399 (85%) | 1045 (0%) | 832 (0%) | 21963 (3%) | 30084 (4%) | 2659 (0%) | 21813 (3%) | 29901 (4%) | 731644 | PASS |
| 64 | 3646 (0%) | 2462 (0%) | 3831625 (95%) | 1889 (0%) | 1084 (0%) | 43893 (1%) | 59205 (1%) | 3902 (0%) | 43766 (1%) | 59007 (1%) | 4050479 | PASS |
| 128 | 6010 (0%) | 8295 (0%) | 21067285 (98%) | 3751 (0%) | 1787 (0%) | 87505 (0%) | 118007 (1%) | 7789 (0%) | 86790 (0%) | 117943 (1%) | 21505162 | PASS |

### GEMM utilization (RTL — meaningful)

| L | QKV mesh | Scores mesh | Attn×V mesh | Out mesh | FFN mesh |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 53% | 35% | 53% | 53% | 64% |
| 32 | 60% | 60% | 61% | 66% | 66% |
| 64 | 56% | 70% | 75% | 57% | 69% |
| 128 | 55% | 75% | 52% | 54% | 67% |

Full exe_act / exe_busy tables: see `BASELINE.md`.

### Wall-clock simulation time (ops only — **not** HW performance)

| L | Wall (s) | Wall (h) |
| ---: | ---: | ---: |
| 16 | 1213 | 0.34 |
| 32 | 2477 | 0.69 |
| 64 | 7403 | 2.06 |
| 128 | 28295 | 7.86 |

An earlier (pre-counters-inline) L=256 Verilator log exists in [`legacy_pre_counters/`](legacy_pre_counters/); it has not been independently confirmed for use with this baseline and must not be merged into the table above. See [`legacy_pre_counters/README.md`](legacy_pre_counters/README.md).

---

## Key findings (short)

1. **Softmax (host) owns the block** — 65% of Verilator cycles at L=16, 98% at L=128; Spike reaches 99.5% at L=256. Doubling L multiplies Softmax cycles by ~5.5–6.7× in these runs.
2. **Residual + RMSNorm** are the other host-scalar costs; absolute cycles grow ~with L; share falls as Softmax grows.
3. **GEMM share** is ~4% at L=16 (Verilator) and **falls below 1%** by L=64+ — Gemmini matmuls are fast relative to host Softmax, not absent.
4. **Correctness clean** — 200/200 Spike, 4/4 Verilator; no open all_negative L=256 margin FAIL on this build.

---

## Memory-footprint checks

| L | int8 tensors | float32 gold tensors | int32 zero-bias tensors | total declared buffers |
| --- | --- | --- | --- | --- |
| 16 | 7424 bytes | 29696 bytes | 6144 bytes | 43264 bytes |
| 32 | 12800 bytes | 51200 bytes | 14336 bytes | 78336 bytes |
| 64 | 26624 bytes | 106496 bytes | 36864 bytes | 169984 bytes |
| 128 | 66560 bytes | 266240 bytes | 106496 bytes | 439296 bytes |
| 256 | 195584 bytes | 782336 bytes | 344064 bytes | 1321984 bytes |

Expected snapshots: `correctness/expected/L*_D16_F64_seed1.h`.

## Notes

- Softmax share of runtime grows with L (host scalar over L×L).
- A `Verilator +max-cycles timeout` is a sim-budget issue, not necessarily a functional FAIL.
- Raw logs: `chipyard/tmp_baseline_validation/phase1_spike/`, `phase2_verilator/`.
