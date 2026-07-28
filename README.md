# Attention Hardware Accelerator

Exploring hardware acceleration for the attention mechanism (QKV projections,
softmax, matmul) on top of Gemmini, UC Berkeley's systolic-array based
DNN accelerator generator, within the Chipyard SoC framework.

## Status

Early stage — building a custom bareMetalC test workload for attention
(Gemmini has no existing QKV/softmax/attention benchmark upstream). 

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

The single-head transformer decoder-block benchmark is recorded separately for
each simulator:

- [Spike functional baseline](baseline-tests/spike/README.md)
- [Verilator cycle-accurate baseline](baseline-tests/verilator/README.md)
