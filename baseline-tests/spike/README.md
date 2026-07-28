# Spike Functional Baseline Test

## Workload

`transformer_block_test` is a single-head transformer decoder block with:

- `SEQ_LEN=16`
- `D_MODEL=16`
- `D_FF=64`
- Gemmini executes all matrix multiplications.
- The host core executes softmax, RMSNorm, and residual additions.

## Command

```shell
cd /home/users/ah072084/chipyard/generators/gemmini/software/gemmini-rocc-tests
spike --extension=gemmini build/bareMetalC/transformer_block_test-baremetal
```

## Result

`PASS`

```text
Q/K/V projections:       952 cycles
Attention scores:        323 cycles
Softmax:               49088 cycles
Attention output:        320 cycles
Output projection:       319 cycles
Residual add 1:         5793 cycles
RMSNorm 1:              6489 cycles
Feed-forward network:    637 cycles
Residual add 2:         5795 cycles
RMSNorm 2:              6490 cycles
Total:                 76206 cycles
```

## Numerical Checks

- `QUANT_SCALE = 1/128`; final tolerance = `0.039` (`5 × QUANT_SCALE`).
- Gold softmax range: `[0.049, 0.078]`.
- Quantized softmax: raw int8 `[6, 10]`.
- Attention scores: raw int8 `[-28, 33]`.
- No tensor reached int8 saturation (`-128` or `127`).

## Conclusions

Spike verifies that the fixed-point workload and Gemmini custom-instruction
sequence are functionally correct. Attention is non-uniform: the score and
softmax ranges retain measurable variation after quantization.

The host scalar softmax consumes 49,088 of 76,206 reported cycles
(approximately 64%). RMSNorm and residual operations also dominate the
reported execution. These are useful software-path profiling observations,
but Spike is not cycle-accurate hardware simulation; do not use its cycle
counts as accelerator performance evidence.
