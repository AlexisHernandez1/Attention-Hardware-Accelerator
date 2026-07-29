# Official Sequence-Length Sweep Baseline (seed=1)

## Setup

- **Official generator:** independent per-tensor PRNG streams from `PRNG_SEED=1`.
- Fixed: `D_MODEL=16`, `D_FF=64`; varied `SEQ_LEN`: 16, 32, 64, 128, 256.
- Spike: float gold + export `expected_final` snapshot.
- Verilator: `SKIP_GOLD=1` + `USE_EXPECTED` (exact int8 match to Spike snapshot).
- Timing from cycle-accurate `GemminiRocketConfig` only; Spike cycles are not RTL performance.
- Legacy shared-PRNG baselines under `baseline-tests/` are historical only — do not mix into before/after tables.

## Verilator Results

Each stage cell is `cycles (percent of total)`.

| L | Total cycles | QKV projections | Attention scores | Softmax | Attention output | Output projection | Residual add 1 | RMSNorm 1 | Feed-forward network | Residual add 2 | RMSNorm 2 | Result |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 16 | 150252 | 2317 (1%) | 637 (0%) | 98346 (65%) | 629 (0%) | 615 (0%) | 9514 (6%) | 13506 (8%) | 1795 (1%) | 9505 (6%) | 13388 (8%) | PASS |
| 32 | 492428 | 2684 (0%) | 1000 (0%) | 393338 (79%) | 909 (0%) | 731 (0%) | 18909 (3%) | 26793 (5%) | 2521 (0%) | 18843 (3%) | 26700 (5%) | PASS |
| 64 | 1763595 | 3443 (0%) | 2390 (0%) | 1577570 (89%) | 1760 (0%) | 988 (0%) | 37707 (2%) | 49344 (2%) | 3724 (0%) | 37544 (2%) | 49125 (2%) | PASS |
| 128 | 6659456 | 5923 (0%) | 8060 (0%) | 6279620 (94%) | 4136 (0%) | 1755 (0%) | 77937 (1%) | 98579 (1%) | 7573 (0%) | 77505 (1%) | 98368 (1%) | PASS |
| 256 | 26004206 | 10250 (0%) | 38062 (0%) | 25248498 (97%) | 11548 (0%) | 3028 (0%) | 140019 (0%) | 198863 (0%) | 17375 (0%) | 138131 (0%) | 198432 (0%) | PASS |

## Memory-Footprint Checks

| L | int8 tensors | float32 gold tensors | int32 zero-bias tensors | total declared buffers |
| --- | --- | --- | --- | --- |
| 16 | 7424 bytes | 29696 bytes | 6144 bytes | 43264 bytes |
| 32 | 12800 bytes | 51200 bytes | 14336 bytes | 78336 bytes |
| 64 | 26624 bytes | 106496 bytes | 36864 bytes | 169984 bytes |
| 128 | 66560 bytes | 266240 bytes | 106496 bytes | 439296 bytes |
| 256 | 195584 bytes | 782336 bytes | 344064 bytes | 1321984 bytes |

Expected snapshots live in `correctness/expected/L*_D16_F64_seed1.h`.

## Notes

- Softmax share of runtime should grow with L (host scalar `expf` over L×L).
- Seed 1 may still show K saturation; that is frozen for this official baseline.
- A `Verilator +max-cycles timeout` means the sim-cycle budget was too small, not necessarily a functional FAIL.
