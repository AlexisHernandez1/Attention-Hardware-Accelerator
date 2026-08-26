# Cross-build comparison and analysis

This document holds **cross-build** tables, ratios, speedups, and narrative
conclusions that combine numbers from more than one Softmax path. Raw measured
results for each build live only in:

| Build | Doc |
| --- | --- |
| Scalar host Softmax (`GemminiRocketConfig`, HW-resadd default) | [`BASELINE.md`](BASELINE.md) |
| HW PWL Softmax (`GemminiPWLSoftmaxConfig`) | [`PWL_SOFTMAX.md`](PWL_SOFTMAX.md) |
| HW I-BERT Softmax (`GemminiIBertSoftmaxConfig`) | [`IBERT_SOFTMAX.md`](IBERT_SOFTMAX.md) |

Numbers below are relocated as published — not recomputed in this pass.

---

## Comparability

Side-by-side with the Verilator numbers already published in [`BASELINE.md`](BASELINE.md)
(§2, `GemminiRocketConfig`, seed=1, HW-resadd default-on), [`PWL_SOFTMAX.md`](PWL_SOFTMAX.md)
(post-tiling-fix Verilator sweep, `GemminiPWLSoftmaxConfig`, seed=1), and
[`IBERT_SOFTMAX.md`](IBERT_SOFTMAX.md) (`GemminiIBertSoftmaxConfig`, seed=1).

| Item | Scalar (`BASELINE.md`) | HW-PWL (`PWL_SOFTMAX.md`) | HW-IBert (`IBERT_SOFTMAX.md`) | Match? |
| --- | --- | --- | --- | --- |
| `D_MODEL` / `D_FF` / `DIM` | 16 / 64 / 16 | 16 / 64 / 16 | 16 / 64 / 16 | yes |
| `PRNG_SEED` / case | 1 / random | 1 / random | 1 / random | yes |
| `USE_HW_RESADD` | default **1** (`tiled_resadd_auto`) | default **1** (not overridden) | default **1** | yes |
| `QUANT_SCALE` / final tol | `1/128` / `5 × QUANT_SCALE` (`0.039`) | same (`QUANT_SCALE=0.008`, tol=`0.039`) | same (`5 × QUANT_SCALE`) | yes |
| Softmax implementation | Host scalar (`expf` + largest-remainder) | HW PWL (`USE_HW_PWL_SOFTMAX=1`) | HW I-BERT (`USE_HW_SOFTMAX=1`) | **intentional difference** |
| Chipyard config | `GemminiRocketConfig` | `GemminiPWLSoftmaxConfig` (`has_pwl_softmax`) | `GemminiIBertSoftmaxConfig` | Softmax RTL differs |

---

## Softmax cycle comparison

Scalar Softmax from [`BASELINE.md`](BASELINE.md); HW-PWL Softmax from
[`PWL_SOFTMAX.md`](PWL_SOFTMAX.md); HW-IBert Softmax from
[`IBERT_SOFTMAX.md`](IBERT_SOFTMAX.md) / `report.tsv`.
Softmax % is Softmax cycles ÷ total for that build. Scalar Softmax dominates
the scalar baseline (**67.2% → 85.9% → 95.1% → 98.1%** of total as L grows).

| L | Scalar Softmax | HW-PWL Softmax | HW-IBert Softmax | PWL ÷ IBert (×) | Scalar Softmax % | PWL Softmax % | IBert Softmax % |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 109903 | 3159 | 2437 | 1.296 | 67.2% | 5.6% | 4.4% |
| 32 | 619397 | 13268 | 9175 | 1.446 | 85.9% | 11.6% | 8.3% |
| 64 | 3831599 | 47958 | 29513 | 1.625 | 95.1% | 19.5% | 13.0% |
| 128 | 21070239 | 227085 | 131411 | 1.728 | 98.1% | 36.4% | 24.9% |

Both hardware Softmax paths are **dramatically faster** than host scalar Softmax
(tens of × at L=16, approaching ~100× Softmax-cycle reduction at L=128). Between
the two HW paths, **I-BERT Softmax uses fewer absolute Softmax cycles than PWL**
at every L (~**1.3–1.7×** fewer Softmax cycles; PWL Softmax ÷ IBert Softmax in
the table). Totals follow the same ordering (IBert total &lt; PWL total &lt;&lt;
scalar total) because Softmax is the stage that moved.

---

## Speedup ratios (scalar ÷ HW-PWL)

Speedups are scalar ÷ HW-PWL. Softmax % is Softmax cycles ÷ total for that build.
Scalar Softmax % values below are recomputed from the same cycle counts as
[`BASELINE.md`](BASELINE.md) (that doc rounds them to one decimal: 67.2% / 85.9% /
95.1% / 98.1%).

| L | Scalar total | HW-PWL total | Total speedup (×) | Scalar Softmax | HW-PWL Softmax | Softmax speedup (×) | Scalar Softmax % | HW-PWL Softmax % |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 163529 | 56663 | 2.885993 | 109903 | 3159 | 34.790440 | 67.2070% | 5.5751% |
| 32 | 720706 | 114429 | 6.298281 | 619397 | 13268 | 46.683524 | 85.9431% | 11.5950% |
| 64 | 4029349 | 245790 | 16.393462 | 3831599 | 47958 | 79.894887 | 95.0923% | 19.5118% |
| 128 | 21480688 | 623942 | 34.427379 | 21070239 | 227085 | 92.785693 | 98.0892% | 36.3952% |
| 256 | — (Spike-only in `BASELINE.md`) | not run on Verilator | — | — | — | — | — | — |

### Share of total after Softmax shrinks

With Softmax cycles much smaller, GEMM / Residual / RMSNorm absolute cycle counts
that stay near the scalar-baseline values (L≤64; see caveat for L=128 Residual
below) necessarily occupy a larger fraction of the reduced total. Exact scalar-side
per-stage cycles and stage-share tables are in [`BASELINE.md`](BASELINE.md)
§2 (“Per-stage cycles” and “Combined stage shares”); pull those numbers directly
rather than re-deriving them here.

Non-Softmax absolute stage cycles (GEMM, Residual, RMSNorm) at L=16/32/64 agree
with [`BASELINE.md`](BASELINE.md)’s per-stage table to within a few percent (Residual /
RMSNorm typically within ~0.5%). At **L=128**, Residual 1/2 absolute cycles are
**not** the same: HW-PWL logs 65301 / 64664 vs scalar baseline 73609 / 72927
(~11% lower on the PWL run); other non-Softmax stages at L=128 remain within a
few percent. That L=128 Residual gap is a **confirmed real divergence** (see
L=128 Residual anomaly below), not a logging glitch — treat Residual absolute counts
at L=128 as non-comparable across configs when reading share shifts; Softmax and
total-cycle columns below are still the measured values from each published run.

---

## Accuracy / ulps comparison

From prior Softmax error characterization (Spike random grids; Step-3 /
`CHAR_SOFTMAX_ERR` work under `chipyard/tmp_baseline_validation/`, summarized
alongside [`SOFTMAX_ERROR_BASELINE.md`](SOFTMAX_ERROR_BASELINE.md)):

| Path | `soft_max_ulps` range | Accuracy (best → worst) |
| --- | --- | --- |
| Scalar host | **[0.608, 0.696]** | best |
| HW PWL | **[1.638, 5.044]** | middle |
| HW I-BERT | **[2.888, 18.741]** | worst of the three |

Ordering: **scalar &lt; PWL &lt; HW I-BERT** (best to worst per-element Softmax
error). All three stay well under the final `5 × QUANT_SCALE` tolerance
(`final_max_over_tol` peaks remain &lt; ~0.5). Frame this as a **speed /
accuracy tradeoff between the two hardware Softmax paths** (I-BERT: fewer
Softmax cycles, higher Softmax ulps; PWL: more Softmax cycles than I-BERT,
tighter Softmax error) — not a PASS/FAIL difference. That tradeoff, plus
PWL being project-owned RTL, is why PWL is pursued as the primary Softmax
deliverable even though I-BERT Verilator PASS is real.

I-BERT Softmax is Gemmini’s **pre-existing** (previously disabled) hardware
path — Normalizer + integer `iexp` in AccumulatorScale — restored and wired
through `GemminiIBertSoftmaxConfig` / `USE_HW_SOFTMAX`, not built from scratch.
PWL Softmax ([`PWL_SOFTMAX.md`](PWL_SOFTMAX.md)) is a **from-scratch** Chisel
unit (`PwlSoftmax` + bridge). That asymmetry is why PWL remains the primary
deliverable even though the I-BERT path “works” on Spike and Verilator: I-BERT
is a reference enablement of upstream Gemmini Softmax HW; PWL is the
project-owned Softmax design.

---

## Non-Softmax stage parity (IBert vs PWL)

At L=16/32/64, QKV / Scores / Attn×V / Out / FFN / Residual / RMSNorm absolute
cycles match [`PWL_SOFTMAX.md`](PWL_SOFTMAX.md) within noise (typically tens of
cycles). The two configs are apples-to-apples outside Softmax itself; Softmax
is the intentional RTL difference (`has_normalizations` vs `has_pwl_softmax`).

---

## L=128 Residual anomaly

| Build | Res1 | Res2 |
| --- | ---: | ---: |
| HW-IBert ([`IBERT_SOFTMAX.md`](IBERT_SOFTMAX.md)) | **65217** | **64712** |
| HW-PWL ([`PWL_SOFTMAX.md`](PWL_SOFTMAX.md)) | 65301 | 64664 |
| Scalar ([`BASELINE.md`](BASELINE.md)) | 73609 | 72927 |

IBert Res1/Res2 match PWL within noise (deltas −84 / +48), and both sit
~**11%** below scalar. This is the **same already-diagnosed anomaly class**
documented below — **not** a new I-BERT-specific bug. Do not re-investigate as
I-BERT-only; treat L=128 Residual absolute counts as non-comparable to scalar
when reading share shifts. Softmax and total-cycle columns remain the measured
values from each published run.

**L=128 Residual investigation (read-only):** Both builds use default
`USE_HW_RESADD=1` (PWL L=128 `rtl_build.log` does not pass `-DUSE_HW_RESADD=0`).
The ~65k Res1 / ~64k Res2 counts appear in **two independent** PWL Verilator
runs (`phase3_l128_dma_debug` and `post_tiling_fix_validation`, identical stage
tables). PWL Res1 at L=128 is ≈ 2× L=64 (65301 vs 2×32780); scalar Res1 is
≈ 2× L=64 **+ ~8.2k** (73609 vs 2×32689). Adjacent post-Softmax GEMM stages
(Attn×V / Out / FFN) match closely at L=128, so the gap is Residual-specific in
the measured window. **Ruled out:** wrong `USE_HW_RESADD` flag; one-off log
corruption; elaboration mismatch on unused CNN knobs
(`has_training_convs` / `has_max_pool`) — matching those to Rocket defaults
(`true`/`true`) and re-running L=128 Verilator left Res1/Res2 at **65301 /
64664** (and Softmax/total unchanged at 227085 / 623942); artifact
`chipyard/tmp_baseline_validation/pwl_l128_residual_elab_ab/SUMMARY.txt`. A
single root cause is still **not** pinned; leading remaining hypothesis is
DRAMSim / host-cache state after ~21M-cycle host Softmax vs ~227k-cycle HW
Softmax before Residual touches `X`.

---

## HW residual vs prior scalar-residual Verilator (Rocket baseline)

Same seed=1, L≤128. Prior path: scalar residual; current: `USE_HW_RESADD=1`.
Raw current totals/stages remain in [`BASELINE.md`](BASELINE.md).

| L | Res1 Δ | Res2 Δ | Total Δ |
| ---: | --- | --- | --- |
| 16 | 11147 → 8387 (**−24.8%**) | 11001 → 8265 (**−24.9%**) | 169122 → 163529 (**−3.3%**) |
| 32 | 21963 → 16489 (**−24.9%**) | 21813 → 16297 (**−25.3%**) | 731644 → 720706 (**−1.5%**) |
| 64 | 43893 → 32689 (**−25.5%**) | 43766 → 32391 (**−26.0%**) | 4050479 → 4029349 (**−0.5%**) |
| 128 | 87505 → 73609 (**−15.9%**) | 86790 → 72927 (**−16.0%**) | 21505162 → 21480688 (**−0.11%**) |

Residual 1/2 each drop ~25% at L=16–64 and ~16% at L=128. Totals move only a few percent (and &lt;0.2% at L=128) because Softmax still owns **67% → 98%** of the block — the stage breakdown in [`BASELINE.md`](BASELINE.md) confirms that interpretation rather than complicating it. Relative residual speedup does **not** grow with L on this RTL sweep (largest %-reduction is at mid L; L=128 residual save is smaller in percent while Softmax’s absolute dominance is larger).

---

## Conclusions

### Scalar baseline (`GemminiRocketConfig`) takeaways

Grounded in the numbers in [`BASELINE.md`](BASELINE.md):

#### 1. Softmax (host scalar) remains the bottleneck at every L

On Verilator seed=1 with HW residual: Softmax is **67.2% → 85.9% → 95.1% → 98.1%** of total as L goes 16 → 32 → 64 → 128. Enabling `tiled_resadd_auto` does not change that ranking.

#### 2. Hardware residual helps Residuals ~16–26%; totals barely move

Res1/Res2 each fall ~25% at L=16–64 and ~16% at L=128 vs the old scalar-residual Verilator table. Totals fall **3.3% → 1.5% → 0.5% → 0.11%** over the same L sweep — Softmax absorbs the win.

#### 3. RMSNorm is now the clearest remaining host-scalar target after Softmax

At L=16, RMSNorm 1+2 is **18.5%** of RTL cycles (larger than Residual 1+2 at **10.2%**). By L=128 both are small versus Softmax, but RMSNorm is still the larger host-scalar block.

#### 4. GEMM share stays small and shrinks with L

Combined GEMM: **4.1% → 1.2% → 0.3% → 0.1%** (Verilator seed=1). Mesh util ~50–75% on short bursts; not the end-to-end clock.

#### 5. Correctness: HW residual matches scalar tensors on the random grid

25/25 Spike PASS under default HW residual; prior on/off compare showed identical Residual/Final int8 ranges on all 25 configs. Verilator 4/4 PASS with matching Softmax stats and no sat banners.

#### Plain-language summary

This baseline is **unmodified Gemmini** for GEMMs **and** residual-add (`tiled_resadd_auto` default on); Softmax and RMSNorm still run on the host. Softmax alone is still about two-thirds of RTL cycles at L=16 and essentially the whole block by L=128 — so hardware residual cuts Residual stage time ~16–26% but moves end-to-end totals only a few percent. Use **Verilator** for any performance claim; use **Spike** (including L=256) for correctness only. Next leverage after Softmax is host RMSNorm (and eventually a real hardware softmax).

### Hardware Softmax paths

- Both HW Softmax paths cut Softmax cycles by tens of × vs host scalar; I-BERT uses fewer Softmax cycles than PWL (~1.3–1.7×) at every L.
- Accuracy ordering soft_max_ulps: scalar &lt; PWL &lt; I-BERT; all stay under final tol. Speed/accuracy tradeoff favors documenting both; PWL remains the primary project-owned Softmax deliverable.
- L=128 Residual absolute cycles for PWL and I-BERT match each other within noise and sit ~11% below scalar — same anomaly class; do not treat as I-BERT-specific.
