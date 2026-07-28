# Sequence-Length Sweep Baseline

## Setup

- Fixed dimensions: `D_MODEL=16`, `D_FF=64`; varied `SEQ_LEN`: 16, 32, 64, 128, 256.
- Every point is force-rebuilt with `TRANSFORMER_CFLAGS`; each first passes Spike before RTL simulation.
- Timing numbers come only from cycle-accurate Verilator using `GemminiRocketConfig`.
- Spike is a functional pre-flight check; its reported cycles are not hardware-performance data.
- Each Verilator log contains timestamped build, Spike, and Verilator status lines.

## Verilator Results

Each stage cell is `cycles (percent of total)`.

| L | Total cycles | QKV projections | Attention scores | Softmax | Attention output | Output projection | Residual add 1 | RMSNorm 1 | Feed-forward network | Residual add 2 | RMSNorm 2 | Result |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 16 | 145744 | 2283 (1%) | 679 (0%) | 97526 (66%) | 663 (0%) | 603 (0%) | 8764 (6%) | 12474 (8%) | 1780 (1%) | 8589 (5%) | 12383 (8%) | PASS |
| 32 | — | — | — | — | — | — | — | — | — | — | — | Verilator +max-cycles timeout |
| 64 | — | — | — | — | — | — | — | — | — | — | — | Verilator +max-cycles timeout |
| 128 | — | — | — | — | — | — | — | — | — | — | — | Verilator +max-cycles timeout |
| 256 | — | — | — | — | — | — | — | — | — | — | — | Verilator +max-cycles timeout |

## Memory-Footprint Checks

| L | int8 tensors | float32 gold tensors | int32 zero-bias tensors | total declared buffers |
| --- | --- | --- | --- | --- |
| 16 | 7424 bytes | 29696 bytes | 6144 bytes | 43264 bytes |
| 32 | 12800 bytes | 51200 bytes | 14336 bytes | 78336 bytes |
| 64 | 26624 bytes | 106496 bytes | 36864 bytes | 169984 bytes |
| 128 | 66560 bytes | 266240 bytes | 106496 bytes | 439296 bytes |
| 256 | 195584 bytes | 782336 bytes | 344064 bytes | 1321984 bytes |

The bare-metal linker script places the image at `0x80000000` but does not declare a stack or DRAM upper bound. `GemminiRocketConfig` uses Rocket Chip's default 256 MiB external-memory window; the table reports the benchmark's declared tensors only.

## Sanity Check

The `L=16` result is expected to match the previously validated manual baseline: `145744` total cycles and `PASS`. A mismatch indicates that the sweep pipeline or simulator configuration needs investigation before comparing larger points.

## Incomplete Points

A `Verilator +max-cycles timeout` result means the simulator's fixed `+max-cycles=10000000` limit expired before the benchmark printed `PASS` or `FAIL`. It is distinct from the script's wall-time ceiling and does not establish numerical correctness or timing for that point.
