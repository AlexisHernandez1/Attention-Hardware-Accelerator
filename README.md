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
- Gemmini was originally built for CNN workloads and has no native RMSNorm hardware. It does ship an experimental, disabled-by-default I-BERT Softmax unit, which we enable and validate as one comparison point. The project's main contribution is a purpose-built PWL Softmax hardware unit, designed for higher accuracy than the existing I-BERT implementation, along with hardware-accelerated residual adds — extending Gemmini's transformer/attention support beyond its default CNN-oriented configuration. RMSNorm currently remains on the host core; a dedicated RMSNorm hardware unit is a planned future extension.
- Simulation-only target (Spike / Verilator) — no FPGA prototyping, so FireSim/FireMarshal are out of scope
## Where the code lives

This repo is the entry point and umbrella for the project — writeup,
notes, and benchmarking scripts that don't belong inside Chipyard's
directory structure. The actual hardware/software changes live in forks
of the upstream projects, pinned to the `attention-accelerator` branch
and chained together via submodules so the whole setup is reproducible
from a single clone:

- **Chipyard fork:** [AlexisHernandez1/chipyard](https://github.com/AlexisHernandez1/chipyard/tree/attention-accelerator) — top-level SoC framework
- **Gemmini fork:** [AlexisHernandez1/Gemmini](https://github.com/AlexisHernandez1/Gemmini/tree/attention-accelerator) — submodule at `generators/gemmini`; the DNN accelerator generator itself, including the PWL/I-BERT Softmax and hardware-resadd changes
- **gemmini-rocc-tests fork:** [AlexisHernandez1/gemmini-rocc-tests](https://github.com/AlexisHernandez1/gemmini-rocc-tests/tree/attention-accelerator) — submodule at `generators/gemmini/software/gemmini-rocc-tests`; bareMetalC test sources, including `transformer_block_test.c`
- **libgemmini fork:** [AlexisHernandez1/libgemmini](https://github.com/AlexisHernandez1/libgemmini/tree/attention-accelerator) — submodule at `generators/gemmini/software/libgemmini`; the Spike extension with Gemmini instruction support (including the I-BERT Softmax plugin path used for the `GemminiIBertSoftmaxConfig` comparison)

Clone the Chipyard fork with `--recurse-submodules` on the
`attention-accelerator` branch to pull in all three of the above forks
at their pinned commits in one step:

```bash
git clone --recurse-submodules -b attention-accelerator \
  https://github.com/AlexisHernandez1/chipyard.git
```

## Background

This project explores RISC-V hardware acceleration for attention, a key computational bottleneck in transformer inference. Built on the open-source Gemmini/Chipyard framework and advised by Professor Tony Wu from the Zhejiang University SPAIL Lab.

## Baseline Tests

**Official baseline:** single-head, `D_MODEL=16`, `D_FF=64`, calibrated
`ACC_SCALE_Q/K`, `SCORE_DEQUANT_SCALE=1/6`, `RMSNORM_GAIN=0.33974210`,
seeds 1–5 × `L∈{16,32,64,128,256}`.

**Setup:** every script under `correctness/scripts/` sources
`$CHIPYARD/env.sh` and expects `$CHIPYARD` to point at your Chipyard
checkout from the previous step. Export it before running anything —
if left unset, the scripts fall back to a hardcoded path from the
original development machine, which will not exist on yours:

```bash
export CHIPYARD=/path/to/your/chipyard
./correctness/scripts/run_attention_baseline_grid.sh
```

**Performance baseline (authoritative):** [`BASELINE.md`](BASELINE.md) —
build conditions (HW GEMM vs host Softmax/RMSNorm/Residual), full Spike 200-config
grid, Verilator L=16–128 seed=1 (raw measured results).

**Hardware PWL Softmax** (`GemminiPWLSoftmaxConfig`): [`PWL_SOFTMAX.md`](PWL_SOFTMAX.md) —
resolved RegInit / fence / DMA-tiling bugs and post-tiling-fix Spike+Verilator validation.

**Hardware I-BERT Softmax** (`GemminiIBertSoftmaxConfig`): [`IBERT_SOFTMAX.md`](IBERT_SOFTMAX.md) —
stock Gemmini I-BERT Softmax/Normalizer path (`Normalizer.scala` + `AccumulatorScale` `iexp`)
enabled via `has_normalizations`, with Spike 25/25 random + 100/100 edge-case validation.

**Cross-build comparison:** [`COMPARISON.md`](COMPARISON.md) —
Softmax cycle / speedup / ulps side-by-sides, L=128 Residual anomaly, and narrative
conclusions across scalar, PWL, and I-BERT builds.

Also: [`correctness/README.md`](correctness/README.md),
[`sweeps/l_sweep/README.md`](sweeps/l_sweep/README.md).

Historical (legacy shared-PRNG / pre-calibration) simulator notes:

- [Spike functional baseline](baseline-tests/spike/README.md)
- [Verilator L=16 (legacy)](baseline-tests/verilator/README.md)
- [Baseline-tests index](baseline-tests/README.md)
- Pre-cal expected headers: [`correctness/expected/legacy_pre_cal/`](correctness/expected/legacy_pre_cal/)

## Hardware/software changes

Actual attention-kernel changes (RMSNorm gain calibration, Q/K quantization
probes, softmax distribution validation) live in the four forks listed under
[Where the code lives](#where-the-code-lives) above. Reproduce them by
cloning the Chipyard fork with `--recurse-submodules` on the
`attention-accelerator` branch, as shown above, then running the baseline
scripts under `correctness/` (see [Baseline Tests](#baseline-tests)).
