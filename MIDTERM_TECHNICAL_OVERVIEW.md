# Attention Hardware Accelerator — Midterm Technical Overview

## Project scope

This project studies **RISC-V acceleration of transformer attention on Gemmini** (UC Berkeley’s systolic-array DNN accelerator) inside **Chipyard**. The umbrella repo is [`attention-hardware-accelerator`](https://github.com/AlexisHernandez1/attention-hardware-accelerator); the working code lives on the `attention-accelerator` branches of Chipyard → Gemmini → `gemmini-rocc-tests`.

**What it is designed to do (as of now):** run a **single-head transformer decoder block** end-to-end in int8, with:

| Stage | Where it runs |
|---|---|
| Q, K, V projections | Gemmini GEMM (`tiled_matmul_auto`) |
| Attention scores \(Q K^\top\) | Gemmini GEMM |
| Softmax | **Host RISC-V** (`quantized_softmax`) |
| Attention × V, output projection | Gemmini GEMM |
| Residual + RMSNorm (×2) | **Host** (`quantized_residual`, `quantized_rmsnorm`) |
| FFN up (ReLU) / down | Gemmini GEMM |

That is the **full single-head decoder block**, not a standalone softmax ASIC or a multi-head LLM kernel. Softmax is explicitly on the host: *“currently no dedicated softmax hardware unit”* (`README.md`).

**Target platform:** **simulation only** — Spike (functional) and Verilator `GemminiRocketConfig` (cycle-accurate). Explicitly **no FPGA**, FireSim, or FireMarshal. There is **no HLS flow**, **no published ASIC/FPGA clock target**, and no in-repo synthesis. Verilator logs report a harness CPU clock of 500 MHz (DRAMSim2 setting), not a design tape-out target.

**Stated research direction** (`chipyard/GEMMINI_ATTENTION_RESEARCH.md`): use Gemmini as a baseline, profile attention, then try a **targeted memory-hierarchy / data-movement change** (Load/Store/Execute controllers, scratchpad, DMA, ROB) — not a ground-up LLM accelerator. That hardware variant **has not been implemented yet**.

Validated shape: `D_MODEL=16`, `D_FF=64`, `L∈{16,32,64,128,256}`, seeds 1–5, `NUM_ATTENTION_HEADS=1`.

---

## Architecture as implemented (not the proposal)

There are **no new attention RTL modules** on `attention-accelerator`. `git diff` vs upstream Gemmini shows **no `.scala` changes** for this work — only README notes and submodule bumps. Pre-existing I-BERT hardware (`Normalizer.scala`, `Activation.SOFTMAX` / `LAYERNORM` / `IGELU`) exists upstream and is used by older `transformers/transformer.c`; **`transformer_block_test.c` does not use it**.

### Software “architecture” (what actually exists)

Primary artifact:

`generators/gemmini/software/gemmini-rocc-tests/bareMetalC/transformer_block_test.c`

**Data / control flow in `main`:**

1. `initialize_tensors()` — PRNG or edge-case fills
2. Optional `build_float_gold()` (skipped under `-DSKIP_GOLD=1`)
3. `gemmini_flush(0)`; optional `counter_configure` for util
4. Timed stages via `read_cycles()`:

```
Q/K/V  --run_gemmini_matmul-->  scores(QKᵀ)
       --quantized_softmax-->  weights
       --Attn×V, W_o-->  residual_1 / RMSNorm_1
       --FFN_up(RELU), FFN_down-->  residual_2 / RMSNorm_2
       --> check_final_output / expected snapshot
```

**GEMM path:** `run_gemmini_matmul` → `tiled_matmul_auto(..., WS)` + `gemmini_fence()`. Working-tree also wraps with `run_gemmini_matmul_hw_util` (Gemmini `EXE_ACTIVE_CYCLE` / `LOOP_MATMUL_ACTIVE_CYCLES` + printf).

**Memory access pattern:** tensors live in host DRAM (`elem_t` arrays). Each GEMM DMA’s tiles into Gemmini’s scratchpad/accumulator via the stock WS loop; there is **no custom FSM, no new AXI/RoCC opcode, no KV-cache hardware**. Softmax/RMSNorm/residual are scalar host loops (`expf`, etc.).

**Fixed-point contract (calibrated defaults):**
`QUANT_SCALE=1/128`, `ACC_SCALE_Q≈0.00596`, `ACC_SCALE_K≈0.00528`, `SCORE_DEQUANT=1/6`, `RMSNORM_GAIN≈0.340`, softmax weights use **largest-remainder** so row sum raw = 128.

There is **no project-owned FSM state machine** beyond Gemmini’s existing controllers. Control flow is ordinary C sequential staging.

### Stages (`enum stage_id`)

| Stage | Name | Implementation |
|-------|------|----------------|
| `STAGE_QKV` | QKV projections | Gemmini GEMM |
| `STAGE_SCORES` | Attention scores QKᵀ | Gemmini GEMM, `transpose_b=true`, `SCORE_SCALE` |
| `STAGE_SOFTMAX` | Softmax | **CPU** `quantized_softmax` |
| `STAGE_ATTENTION` | Attn × V | Gemmini GEMM |
| `STAGE_OUTPUT_PROJECTION` | × W_o | Gemmini GEMM |
| `STAGE_RESIDUAL_1` | Residual add 1 | **CPU** `quantized_residual` |
| `STAGE_RMSNORM_1` | RMSNorm 1 | **CPU** `quantized_rmsnorm` |
| `STAGE_FFN` | FFN up+down | Gemmini GEMM (`RELU` on up) |
| `STAGE_RESIDUAL_2` | Residual add 2 | **CPU** `quantized_residual` |
| `STAGE_RMSNORM_2` | RMSNorm 2 | **CPU** `quantized_rmsnorm` |

### Edge cases (`EDGE_CASE_ID`)

`initialize_tensors` + `check_edge_case_assertions`: random, all_zeros, all_ones, all_max_mag, one_hot, checkerboard, all_negative, near_zero.

---

## Work completed (chronological)

### Upstream context (not this project’s commits)

- ~2022: Gemmini I-BERT Softmax/LayerNorm/iGELU RTL + `transformers/` software
- Chipyard configs such as `Int8TransformerGemminiRocketConfig` (configs only)

### This project

| When | Where | What |
|---|---|---|
| **2026-07-17** | Gemmini `5250d31` *Attention Hardware Accelerator Project Plan*; rocc-tests `af0d2de` *Add single-head transformer decoder-block baseline test* | First `transformer_block_test.c`; umbrella README |
| **2026-07-27–28** | umbrella `ec92976`, `470d6a1` | Baseline Spike/Verilator writeups; L-sweeps; observed Softmax \(O(L^2)\) and saturation |
| **2026-07-29** | `73f5d3a` *Add RMSNorm gain calibration and softmax distribution probe* | Frozen `ACC_SCALE_Q/K`, `SCORE_DEQUANT`, `RMSNORM_GAIN`; softmax ΔH probes |
| **2026-07-29** | `b347b88` *Promote edge-case suite… and fix softmax weight mass* | `EDGE_CASE_ID` 0–7; largest-remainder softmax (fixes L=256 uniform mass doubling) |
| **2026-07-29** | umbrella docs / `37aefb5` | `correctness/README.md`, `WHAT_THIS_PROVES.md`, expected headers, 200-config grid |

**Verified so far (documented):**

- Spike grid **200/200 PASS** (8 cases × 5 seeds × 5 L) via `correctness/scripts/run_attention_baseline_grid.sh`
- Softmax mass fix: edge grid **171/175 → 175/175** after largest-remainder (`WHAT_THIS_PROVES.md`)
- Verilator seed=1 L-sweep (counters-inline baseline): L=16…128 **PASS**; Softmax share rises with L. L=256 under this build was **not** completed (incomplete mid-Softmax attempt purged).
- Pre-counters-inline L=256 Verilator artifacts (older util-printf-era logs; contain a PASS marker / ~127M cycles / ~125 890 s wall in-file, **not independently re-verified**) are archived under `sweeps/l_sweep/legacy_pre_counters/` — historical only, not part of the current L=16–128 table. Separately, an older Jul 28 `L256_D16_F64_seed1.status.log` also claimed PASS (~5.4 h wall).

**Uncommitted (working tree):** HW util counters + `DBG_SOFTMAX_BRANCHES` in `transformer_block_test.c` (+184/−23 vs `b347b88`).

### Companion umbrella repo commits (selected)

| Hash | Date | Message |
|------|------|---------|
| `9a1ed2c` | 2026-07-17 | Add project README |
| `ec92976` | 2026-07-27 | Add baseline tests and sweeps |
| `470d6a1` | 2026-07-28 | L sweeps; notes O(L²) + saturation |
| `9bd7f5d`–`72a39f6`–`93f1f10` | 2026-07-29 | Correctness docs, calibrated expected headers, fork links |
| `37aefb5` | 2026-07-29 | Edge-case grid + document softmax mass fix |

### Correctness claims (from `correctness/WHAT_THIS_PROVES.md`)

1. **RMSNorm gain retune** (`RMSNORM_GAIN=0.33974210`) keeps residual/RMSNorm in int8 safely (25/25 random gold PASS, 0/25 saturation).
2. **Hardware Q/K int8 scoring** preserves attention distribution vs unclipped float gold (worst ΔH_norm 0.015 < 0.03).
3. **End-to-end block** matches Spike float gold / expected snapshots under calibrated defaults.
4. **Named edge cases** (IDs 1–7) pass case-specific assertions (175/175 → 200/200 with random).
5. **Softmax weight mass fix:** largest-remainder renormalization so Σ raw weights = 128 (critical at L=256 uniform).

---

## Roadblocks and open issues

No `TODO`/`FIXME` markers in the project’s new sources. Real friction shows up in **reruns and diagnostics**:

1. **Softmax on the host dominates Verilator time** (~L²). Confirmed counters-inline Verilator: L=128 ~8 h wall. L=256 under the current build was **not** rerun to completion after a mid-Softmax host shutdown; expect ~32–40 h from the L=16–128 trend if/when it is. A pre-counters-inline L=256 log set is archived at `sweeps/l_sweep/legacy_pre_counters/` (unconfirmed; see that README) and must not be merged into the current baseline table.

2. **Int8 softmax mass bug (resolved, multi-attempt):** independent round-half-up at L=256 uniform rows made `128/L=0.5` → every weight 1 → Σ≈2 → Residual-1 blow-up (`all_negative`). Fixed in `quantized_softmax` (commit `b347b88`); archive under `correctness/archive/diagnostics/`.

3. **Q/K / RMSNorm calibration was iterative:** magnitude sweeps (`correctness/magnitude_sweep/`), QK_W=28 vs 56, gain retune, legacy path `-DUSE_LEGACY_QK_SCALES=1`. Saturation detector asymmetry (software clip ±127 vs `elem_t` −128/127) required a later detector fix.

4. **Shared-PRNG made K saturation look L-dependent** — replaced with independent streams (`mix_seed` / `fill_random`); legacy baselines deprecated (`baseline-tests/README.md`).

5. **Spike PK demand paging** vs large workloads documented in `GEMMINI_ATTENTION_RESEARCH.md` (`pk -p`); environment pitfall, not RTL.

6. **Wall-clock vs cycle instrumentation:** enabling `run_gemmini_matmul_hw_util` roughly **2×** reported stage cycles; always-on `check_softmax_vs_gold` / `print_raw_acc_stats` sit **outside** stage timers but still cost Verilator time. README L-sweep tables can drift from latest unattended logs.

7. **No custom memory-hierarchy RTL yet** — the plan’s core hardware step is still open.

---

## What’s not yet done (vs proposal / README)

From `GEMMINI_ATTENTION_RESEARCH.md` evaluation sequence and root `README.md`:

| Planned / implied | Status |
|---|---|
| Define attention workload & profile baseline | **Done** (software baseline + Spike/Verilator) |
| Hypothesize & implement Gemmini memory-system variant | **Not started** (no Scala/RTL delta) |
| Integrate controllers / ROB / DMA / ISA as needed | **Not started** |
| Compare baseline vs variant (latency, traffic, util) | **Not started** (only baseline profiling) |
| Dedicated softmax (or attention) hardware unit | Explicitly **out of scope for now** |
| FPGA / FireSim bring-up | Explicitly **out of scope** |
| Multi-head, L>256, other D/D_FF shapes | Explicitly **out of scope** |
| `sweeps/width_sweep/` (D_MODEL 16→256) | Script only; **no results** |
| Official power / synthesis utilization | Not claimed (“no synthesis”) |
| Commit / stabilize HW-util instrumentation | Dirty tree; README timing tables need refresh |

### Explicit out of scope (from docs)

- Multi-head attention
- Non-D16/F64 shapes, L>256
- Standalone FFN correctness claims
- Silicon bring-up / power as official results
- FireSim / FireMarshal / FPGA prototyping

---

## Key paths

| Role | Path |
|------|------|
| Umbrella README | `attention-hardware-accelerator/README.md` |
| Correctness claims | `attention-hardware-accelerator/correctness/WHAT_THIS_PROVES.md` |
| Correctness suite | `attention-hardware-accelerator/correctness/README.md` |
| Research plan | `chipyard/GEMMINI_ATTENTION_RESEARCH.md` |
| Baseline test | `chipyard/generators/gemmini/software/gemmini-rocc-tests/bareMetalC/transformer_block_test.c` |
| Makefile hooks | `.../bareMetalC/Makefile` |
| Official L-sweep notes | `attention-hardware-accelerator/sweeps/l_sweep/README.md` |
| Softmax-mass diagnostics archive | `attention-hardware-accelerator/correctness/archive/diagnostics/` |
| Older HW-norm path (not used by baseline) | `.../transformers/transformer.c` |

### Linked forks

- Chipyard: `AlexisHernandez1/chipyard` — branch `attention-accelerator`
- Gemmini: `AlexisHernandez1/Gemmini` — branch `attention-accelerator`
- gemmini-rocc-tests: `AlexisHernandez1/gemmini-rocc-tests` — branch `attention-accelerator`

---

## One-line midterm status

The project has a **solid single-head int8 decoder-block correctness and profiling baseline on stock Gemmini**, with Softmax identified as the main cycle bottleneck; it has **not yet modified Gemmini’s memory hierarchy or added attention-specific RTL**, which is where the research plan says the accelerator contribution should go.

---

*Generated for midterm reporting from repository docs, commit history, and implemented source as of 2026-07-31. Does not regenerate or replace `sweeps/l_sweep/README.md`.*
