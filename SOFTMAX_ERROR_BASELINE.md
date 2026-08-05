# Softmax Quantization Error Baseline (Step 1)

**Status:** Step 1 of 4 in the hardware-softmax tolerance derivation plan.  
**Role:** Empirical baseline of float→int8 softmax quantization error on the current scalar path — not a performance number and not a HW-softmax acceptance target.  
**Companion:** Performance / cycle baseline remains [`BASELINE.md`](BASELINE.md). Artifacts: [`results/softmax_error_baseline/`](results/softmax_error_baseline/).

---

## What this measures

Float-gold softmax vs. the current scalar path’s int8-quantized softmax output, **same logits**.

- Gold: row-wise float `softmax(dequant_score(scores))` on the Gemmini score matrix that fed `quantized_softmax`.
- DUT: `dequant(attention_weights)` after largest-remainder int8 quantization under `QUANT_SCALE`.
- Isolates **pure float→int8 quantization rounding error**. No hardware approximation is involved yet.

Instrumentation: `-DCHAR_SOFTMAX_ERR=1` in `transformer_block_test` (`SOFTMAX_QUANT_ERR` / `FINAL_GOLD_ERR` log lines). Sweep driver (disposable scratch): `chipyard/tmp_baseline_validation/run_softmax_err_char.sh`.

---

## Sweep coverage

| Item | Value |
| --- | --- |
| Grid | random (`EDGE_CASE_ID=0`) × `L ∈ {16, 32, 64, 128, 256}` × `PRNG_SEED ∈ {1…5}` |
| Configs | **25** |
| Result | **25 / 25 PASS** |
| Histogram parse | **25 / 25** logs matched `hist_ulps=...` (no silent empty aggregate) |
| Simulator | Spike + Gemmini functional model (correctness path) |

---

## Results

### Softmax element error (vs float gold, same logits)

Pooled over all parsed configs (`total elems = 436480`):

| `|err|` bin (ulps of `QUANT_SCALE`) | Count | Share |
| --- | ---: | ---: |
| `[0–0.5)` | 398780 | **91.36%** |
| `[0.5–1)` | 37700 | **8.64%** |
| `[1–2)` and above | 0 | **0%** |

- **Max absolute error** (`soft_max`) ≈ **0.005** in probability units — consistent across all 25 configs.
- `soft_max_ulps` range **[0.608, 0.696]** (still under 1 ulp of `QUANT_SCALE`).
- `soft_mae` ≈ **0.002** on every config (print precision).

### Downstream final output (vs gold, existing check)

Tolerance unchanged: **`5 × QUANT_SCALE`**.

| Metric | Range / count |
| --- | --- |
| `final_max_over_tol` | **0.29 – 0.46** |
| Configs with `n_over_tol > 0` | **0** |

---

## Interpretation

Today’s scalar quantization pipeline uses **well under half** its final-output error budget (`final_max_over_tol` peaks at ~0.46). That leaves headroom for a hardware approximation to introduce additional error while still passing the final check.

This is a **baseline measurement, not a target**. A future hardware softmax unit is **not** required to match the 0.005 `soft_max` figure; it only needs **total propagated error** to stay within the `5 × QUANT_SCALE` final tolerance. Steps 3–4 of the tolerance-derivation plan will determine what approximation budget that implies.

---

## Artifacts

Full per-config data under [`results/softmax_error_baseline/`](results/softmax_error_baseline/):

| Path | Contents |
| --- | --- |
| `report.tsv` | Per-(L, seed) soft/final metrics |
| `SUMMARY.txt` | Pooled histogram + aggregate ranges |
| `logs/` | Per-config Spike + build logs |
| `sweep_stdout.log` | Full sweep console transcript |

---

## Status in the tolerance plan

| Step | Content | Status |
| ---: | --- | --- |
| **1** | Scalar float→int8 softmax quant error (this document) | **Done** |
| **2** | Approximation-method error vs float gold (fixed-point exp / HW model) | Blocked — needs a fixed model |
| **3** | Downstream propagation of approx error into final output | Blocked on Step 2 |
| **4** | Recommend tolerance / gate for `USE_HW_SOFTMAX` | Blocked on Steps 2–3 |

Do **not** invent a PWL/LUT or widen `FINAL_TOLERANCE` to force a pass. Steps 2–4 wait on a real (or explicitly agreed) fixed-point exp approximation before a HW-softmax tolerance is recommended.
