# Legacy pre-calibration expected headers

These Spike-exported `expected_final` snapshots were generated under the
**pre-calibration** baseline (`ACC_SCALE_Q/K = QUANT_SCALE`,
`SCORE_DEQUANT_SCALE = QUANT_SCALE`, `RMSNORM_GAIN = 0.4`).

They are **superseded** by the official calibrated baseline headers in
`correctness/expected/L*_D16_F64_seed1.h`. Kept only for historical diff /
A/B comparison with `-DUSE_LEGACY_QK_SCALES=1`. Do not use for Verilator
regression of the current baseline.
