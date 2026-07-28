# Verilator Cycle-Accurate Baseline Test

## Workload

`transformer_block_test` is a single-head transformer decoder block with:

- `SEQ_LEN=16`
- `D_MODEL=16`
- `D_FF=64`
- Gemmini executes all matrix multiplications.
- The host core executes softmax, RMSNorm, and residual additions.

## Command

```shell
cd /home/users/ah072084/chipyard/sims/verilator
make CONFIG=GemminiRocketConfig run-binary \
  BINARY=/home/users/ah072084/chipyard/generators/gemmini/software/gemmini-rocc-tests/build/bareMetalC/transformer_block_test-baremetal
```

## Result

`PASS`

```text
Q/K/V projections:      2283 cycles
Attention scores:        679 cycles
Softmax:               97526 cycles
Attention output:        663 cycles
Output projection:       603 cycles
Residual add 1:         8764 cycles
RMSNorm 1:             12474 cycles
Feed-forward network:   1780 cycles
Residual add 2:         8589 cycles
RMSNorm 2:             12383 cycles
Total:                145744 cycles
```

## Numerical Checks

- `QUANT_SCALE = 1/128`; final tolerance = `0.039` (`5 × QUANT_SCALE`).
- Gold softmax range: `[0.049, 0.078]`.
- Quantized softmax: raw int8 `[6, 10]`.
- Attention scores: raw int8 `[-28, 33]`.
- No tensor reached int8 saturation (`-128` or `127`).

## Conclusions

This passing execution is the cycle-accurate `GemminiRocketConfig` baseline
for future architecture changes. Its 145,744 cycles include Rocket-core work,
Gemmini command and DMA overhead, and memory-system effects, so it is suitable
for end-to-end before/after comparisons at the same configuration.

Softmax accounts for 97,526 cycles (approximately 67% of total), while the two
RMSNorm stages add 24,857 cycles (approximately 17%). Together, the
scalar-host stages are the primary baseline bottleneck. A Gemmini memory or
dataflow change should therefore be evaluated separately for GEMM and
end-to-end impact: accelerating matrix multiplication alone cannot remove
most of this workload's current runtime.

The benchmark's reported GEMM utilization is an ideal-MAC estimate divided by
whole-stage cycles, not direct systolic-array occupancy.
