# Correctness workflow: Spike gold + Verilator expected snapshots

This project uses two different checks on purpose.

## Simple picture

1. **Spike (many seeds)** — generate random-looking inputs from a seed, recompute
   the block in float (`build_float_gold`), run Gemmini, compare → catch bugs.
2. **Export** — if Spike `PASS`es, save the final int8 tensor as a header.
3. **Verilator** — same seed, **skip** float gold, compare Gemmini’s final int8
   to that saved tensor → fast RTL timing + regression check.

Spike hunts bugs across data. Verilator proves hardware matches a known-good
answer and reports real cycles.

## Compile-time knobs (`TRANSFORMER_CFLAGS`)

| Flag | Meaning |
| --- | --- |
| `-DPRNG_SEED=<n>` | Which reproducible random case (default `1`) |
| `-DSKIP_GOLD=1` | Do not run `build_float_gold` on the board |
| `-DUSE_EXPECTED` | Compare to `transformer_expected.h` (exact int8) |
| `-DDUMP_EXPECTED=1` | After PASS, print a blob the export script parses |
| `-DDBG_ABS_CYCLES=1` | Absolute cycle markers (optional debug) |

`SKIP_GOLD=1` **requires** `-DUSE_EXPECTED` (compile error otherwise).

## Independent tensor seeds (important)

Each of `X`, `W_q`, `W_k`, … is filled from its **own** stream derived from
`PRNG_SEED`. Changing `SEQ_LEN` no longer reshuffles `W_k`. Same seed ⇒ same
weights at L=16 and L=32 (X just has more rows). That fixes the old false
“K only saturates at L≥32” effect from a single shared PRNG.

## Scripts

All under `correctness/scripts/` (from this repo root):

```bash
# 1) Multi-seed Spike + float gold (find fragile / saturating cases)
./correctness/scripts/spike_seed_sweep.sh 16 16 64 1 8

# 2) Freeze one trusted seed as an expected header
./correctness/scripts/spike_export_expected.sh 16 16 64 1
./correctness/scripts/spike_export_expected.sh 32 16 64 1

# 3) Verilator with snapshot (no on-device gold)
./correctness/scripts/verilator_run_expected.sh 16 16 64 1
./correctness/scripts/verilator_run_expected.sh 32 16 64 1 50000000
```

Artifacts:

- `correctness/expected/L*_D*_F*_seed*.h` — generated snapshots  
- `correctness/expected/include_*/transformer_expected.h` — include path for builds  
- `correctness/logs/` — Spike/Verilator logs  

## Saturation and varied seeds

- Some seeds can still saturate K (wide Q/K init ranges). That is a property of
  **that seed’s weights**, not of L by itself anymore.
- Varied Spike seeds help you **discover** saturating vs clean cases.
- For fair before/after hardware timing, pick one seed (often `1`), note whether
  it saturates, and keep using it. Do not mix seeds when comparing cycle counts.

## Recreating bugs

Yes — fully. Record:

- `SEQ_LEN`, `D_MODEL`, `D_FF`
- `PRNG_SEED`
- Whether gold or expected mode
- Commit hashes of Chipyard/Gemmini/test

Same flags ⇒ same inputs ⇒ same failure. Export the expected header only after
Spike PASS so Verilator regressions replay the trusted case.

## What Verilator does *not* do in expected mode

It does not re-roll random data or re-run float gold. Diversity stays on Spike
(`spike_seed_sweep.sh`). Re-export expected headers after intentional numeric
software changes, or you will get false FAILs.

## Relation to old L=16 / L=32 baselines

Independent seeding **changes** the random tensors relative to the original
shared-PRNG baselines. After exporting new seed=`1` snapshots, treat new
Verilator numbers as the regression baseline for that seed, or keep documenting
legacy runs separately under `baseline-tests/`.
