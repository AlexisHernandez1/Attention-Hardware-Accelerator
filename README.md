# Attention Hardware Accelerator

Exploring hardware acceleration for the attention mechanism (QKV projections,
softmax, matmul) on top of Gemmini, UC Berkeley's systolic-array based
DNN accelerator generator, within the Chipyard SoC framework.

## Status

Early stage — single-head transformer decoder-block bareMetalC baseline on
Gemmini (Spike / Verilator). Calibrated Q/K ACC scales, score dequant, and
RMSNorm gain are the **official defaults**; residual/RMSNorm and softmax ΔH
assertions are baked into the test. See
[`correctness/README.md`](correctness/README.md) and
[`correctness/WHAT_THIS_PROVES.md`](correctness/WHAT_THIS_PROVES.md).

## Approach

- Softmax occurs on the host core (currently no dedicated softmax hardware unit)
- Focus is on improving memory/data movement to keep the matmul phases fed
- Simulation-only target (Spike / Verilator) — no FPGA prototyping, so
  FireSim/FireMarshal are out of scope

## Where the code lives

This repo is the entry point and umbrella for the project — writeup,
notes, and benchmarking scripts that don't belong inside Chipyard's
directory structure. The actual hardware/software changes live in forks
of the upstream projects, pinned to specific commits so the setup is
reproducible:

- **Chipyard fork:** [AlexisHernandez1/chipyard](https://github.com/AlexisHernandez1/chipyard/tree/attention-accelerator)
- **Gemmini fork:** [AlexisHernandez1/Gemmini](https://github.com/AlexisHernandez1/Gemmini/tree/attention-accelerator)

## Background

This project explores RISC-V hardware acceleration for attention, a key computational bottleneck in transformer inference. Built on the open-source Gemmini/Chipyard framework and advised by Professor Tony Wu from the Zhejiang University SPAIL Lab.

## Baseline Tests

**Official baseline:** single-head, `D_MODEL=16`, `D_FF=64`, calibrated
`ACC_SCALE_Q/K`, `SCORE_DEQUANT_SCALE=1/6`, `RMSNORM_GAIN=0.33974210`,
seeds 1–5 × `L∈{16,32,64,128,256}`.

```bash
./correctness/scripts/run_attention_baseline_grid.sh
```

**Performance baseline (authoritative):** [`BASELINE.md`](BASELINE.md) —
build conditions (HW GEMM vs host Softmax/RMSNorm/Residual), full Spike 200-config
grid, Verilator L=16–128 seed=1, and bottleneck analysis.

Also: [`correctness/README.md`](correctness/README.md),
[`sweeps/l_sweep/README.md`](sweeps/l_sweep/README.md).

Historical (legacy shared-PRNG / pre-calibration) simulator notes:

- [Spike functional baseline](baseline-tests/spike/README.md)
- [Verilator L=16 (legacy)](baseline-tests/verilator/README.md)
- [Verilator L=32 (legacy)](baseline-tests/verilator-L32/README.md)
- [Baseline-tests index](baseline-tests/README.md)
- Pre-cal expected headers: [`correctness/expected/legacy_pre_cal/`](correctness/expected/legacy_pre_cal/)

## Hardware/software changes

Actual attention-kernel changes (RMSNorm gain calibration, Q/K quantization
probes, softmax distribution validation) live in a fork of Chipyard, chained
through its submodules:

- **Chipyard fork**: https://github.com/AlexisHernandez1/Chipyard — branch `attention-accelerator`
- **Gemmini fork**: https://github.com/AlexisHernandez1/Gemmini — branch `attention-accelerator`
- **Gemmini-rocc-tests fork**: https://github.com/AlexisHernandez1/gemmini-rocc-tests — branch `attention-accelerator`

Clone the Chipyard fork (with `--recurse-submodules`) on the `attention-accelerator`
branch to reproduce the hardware-side results referenced in this repo's writeups
and probe outputs under `correctness/`.
