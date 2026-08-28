# Attention Hardware Accelerator

Hardware acceleration for transformer attention (QKV projections, softmax,
matmul) built on top of [Gemmini](https://github.com/ucb-bar/gemmini), UC
Berkeley's systolic-array DNN accelerator generator, within the
[Chipyard](https://github.com/ucb-bar/chipyard) SoC framework. Gemmini was
built for CNN workloads and ships no RMSNorm hardware and only an
experimental, disabled-by-default I-BERT Softmax unit; this project adds a
purpose-built **hardware PWL Softmax unit** (validated as more accurate than
the existing I-BERT path) and **hardware-accelerated residual adds**, and
separately enables/validates Gemmini's existing I-BERT Softmax as a
comparison point. Everything is simulation-only (Spike functional / Verilator
cycle-accurate) — no FPGA prototyping, so FireSim/FireMarshal are out of
scope. See [`BASELINE.md`](BASELINE.md), [`PWL_SOFTMAX.md`](PWL_SOFTMAX.md),
[`IBERT_SOFTMAX.md`](IBERT_SOFTMAX.md), and [`COMPARISON.md`](COMPARISON.md)
for full results.

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

```
Attention-Hardware-Accelerator   (this repo — writeups, scripts, results)
        ↓ references
chipyard        (fork, branch: attention-accelerator)
  └─ generators/gemmini              → Gemmini fork            (branch: attention-accelerator)
       └─ software/gemmini-rocc-tests → gemmini-rocc-tests fork (branch: attention-accelerator)
       └─ software/libgemmini         → libgemmini fork         (branch: attention-accelerator)
```

## Getting Started

### 0. Prerequisites

Chipyard's standard system dependencies (Java, Verilator, RISC-V toolchain
deps, Conda, etc.) — install these first, per upstream Chipyard's docs:
https://chipyard.readthedocs.io/en/latest/Chipyard-Basics/Initial-Repo-Setup.html#default-requirements-installation

Disk/time budget: the full install + build is multi-hour; a Verilator run at
L=128 alone takes several hours of wall time on the hardware used for this
project's published numbers.

### 1. Clone

```bash
git clone --recurse-submodules -b attention-accelerator \
  https://github.com/AlexisHernandez1/chipyard.git
cd chipyard
```

`--recurse-submodules` pulls `generators/gemmini` (pinned commit), which in
turn pulls `software/gemmini-rocc-tests` and `software/libgemmini` (also
pinned). If you cloned without it:

```bash
git submodule update --init --recursive
```

Separately, clone this repo (writeups/scripts — not a submodule of chipyard):

```bash
cd ..
git clone https://github.com/AlexisHernandez1/Attention-Hardware-Accelerator.git
```

### 2. Install Chipyard + Spike + Gemmini software

From inside the `chipyard` clone:

```bash
./build-setup.sh
source env.sh
```

This project targets **simulation only** (Spike + Verilator) — when
`build-setup.sh` prompts for which toolchains/flows to install, you can skip
FireSim/FireMarshal.

Then install the Gemmini software runtime and build the test binaries:

```bash
cd generators/gemmini
make -C software/libgemmini install

cd software/gemmini-rocc-tests
./build.sh
```

RISC-V binaries land in `build/` (bareMetal / linux / pk variants).

### 3. Set the `CHIPYARD` environment variable

Every script under `correctness/scripts/` and `sweeps/` in this repo expects
a `CHIPYARD` env var pointing at your local chipyard clone. If unset, they
fall back to a hardcoded path from the original development machine and will
fail. Set this once per shell (add it to your shell profile if you'll be
running these repeatedly):

```bash
export CHIPYARD=/absolute/path/to/your/chipyard
```

### 4. Build a Verilator simulator for a given config

Three configs exist, one per hardware variant, defined in
`chipyard/GemminiConfigs.scala` inside the Gemmini fork:

| Config | What it builds |
| --- | --- |
| `GemminiRocketConfig` | Baseline — GEMM + HW residual-add (`tiled_resadd_auto`); Softmax and RMSNorm on host scalar C |
| `GemminiPWLSoftmaxConfig` | Adds the project's custom hardware PWL Softmax unit |
| `GemminiIBertSoftmaxConfig` | Enables Gemmini's existing (upstream, disabled-by-default) I-BERT Softmax hardware, as a comparison point |

Build (from `chipyard/sims/verilator`), substituting the config you want:

```bash
cd $CHIPYARD/sims/verilator
make CONFIG=GemminiRocketConfig
# or: make CONFIG=GemminiPWLSoftmaxConfig
# or: make CONFIG=GemminiIBertSoftmaxConfig
```

This elaborates the RTL and produces `simulator-chipyard-<Config>` in that
directory — expect this to take a while.

> **`make clean` is scoped per-config** (`CONFIG=X clean` only removes that
> config's build). Don't run an unscoped clean across configs unless you
> intend to wipe all of them.

### 5. Run a single test binary under Verilator

Baseline binaries (`transformer_block_test` and friends) come from
`gemmini-rocc-tests/bareMetalC`, already built in step 2:

```bash
cd $CHIPYARD/sims/verilator
make CONFIG=<Config> run-binary \
  BINARY=$CHIPYARD/generators/gemmini/software/gemmini-rocc-tests/build/bareMetalC/transformer_block_test-baremetal
```

Relevant compile-time flags for `transformer_block_test.c` (set via the
Makefile's `TRANSFORMER_CFLAGS`, then rebuild the binary before running it
under Verilator):

| Flag | Effect |
| --- | --- |
| `-DUSE_HW_RESADD=0` | Use scalar residual-add instead of the HW default (A/B comparison only) |
| `-DUSE_HW_PWL_SOFTMAX=1 -DHAS_PWL_SOFTMAX` | Route Softmax through the hardware PWL unit (requires `GemminiPWLSoftmaxConfig`) |
| `-DUSE_HW_SOFTMAX=1` | Route Softmax through Gemmini's I-BERT hardware (requires `GemminiIBertSoftmaxConfig`) |
| `-DPRNG_SEED=<n>` | Seed for random input generation |
| `-DEDGE_CASE_ID=<0..7>` | 0 = random baseline; 1–7 = named edge-case fills (zeros/ones/max-mag/one-hot/checkerboard/all-negative/near-zero) |
| `-DUSE_EXPECTED` | Compare against a pre-exported snapshot instead of computing gold on-device (needed for Verilator) |

Example, an L=32 PWL-softmax run:

```bash
cd $CHIPYARD/generators/gemmini/software/gemmini-rocc-tests
make TRANSFORMER_CFLAGS="-DSEQ_LEN=32 -DUSE_HW_PWL_SOFTMAX=1 -DHAS_PWL_SOFTMAX -DPRNG_SEED=1 -DUSE_EXPECTED" \
  bareMetalC/transformer_block_test-baremetal

cd $CHIPYARD/sims/verilator
make CONFIG=GemminiPWLSoftmaxConfig run-binary \
  BINARY=$CHIPYARD/generators/gemmini/software/gemmini-rocc-tests/build/bareMetalC/transformer_block_test-baremetal
```

For Spike (fast functional correctness, not cycle-accurate — regression
only, never performance numbers):

```bash
spike --extension=gemmini $CHIPYARD/generators/gemmini/software/gemmini-rocc-tests/build/bareMetalC/transformer_block_test-baremetal
```

### 6. Run the official correctness suite (Spike)

From this repo, with `CHIPYARD` set (step 3):

```bash
./correctness/scripts/run_attention_baseline_grid.sh
# subset of seeds:      ./correctness/scripts/run_attention_baseline_grid.sh 1 2
# resume after a crash: ./correctness/scripts/run_attention_baseline_grid.sh --resume
```

Runs the full official grid: random baseline + all 7 edge cases, seeds 1–5 ×
`L ∈ {16,32,64,128,256}`, `D_MODEL=16`, `D_FF=64`. Results in
`correctness/logs/baseline_grid/summary.tsv`; PASS criteria are documented in
[`correctness/README.md`](correctness/README.md).

To (re-)export expected-value headers for Verilator (which has no on-device
float gold):

```bash
./correctness/scripts/export_baseline_expected.sh
# single point: ./correctness/scripts/spike_export_expected.sh 16 16 64 1
```

Then run under Verilator against a snapshot:

```bash
./correctness/scripts/verilator_run_expected.sh 16 16 64 1
```

### 7. Run the L-sweep (Verilator, performance)

```bash
./sweeps/l_sweep/run.sh
# or, for a long unattended run:
./sweeps/l_sweep/run_verilator_seed1_unattended.sh
```

Reproduces the seed=1, `L ∈ {16,32,64,128}` cycle-count sweep referenced in
`BASELINE.md`, `PWL_SOFTMAX.md`, and `COMPARISON.md`. Budget real time — L=128
alone took several hours on the original hardware. **L=256 has never
completed under Verilator for any config** — Spike (functional,
non-cycle-accurate) validates cleanly at L=256, but Verilator runs are only
published through L=128.

## Results

**Official baseline:** single-head, `D_MODEL=16`, `D_FF=64`, calibrated
`ACC_SCALE_Q/K`, `SCORE_DEQUANT_SCALE=1/6`, `RMSNORM_GAIN=0.33974210`,
seeds 1–5 × `L∈{16,32,64,128,256}`. Read these in order if you're new to the
project:

1. [`correctness/README.md`](correctness/README.md) + [`correctness/WHAT_THIS_PROVES.md`](correctness/WHAT_THIS_PROVES.md) — what "PASS" means and why
2. [`BASELINE.md`](BASELINE.md) — authoritative performance baseline: build conditions (HW GEMM vs host Softmax/RMSNorm/Residual), full Spike 200-config grid, Verilator L=16–128 seed=1
3. [`PWL_SOFTMAX.md`](PWL_SOFTMAX.md) — hardware PWL Softmax (`GemminiPWLSoftmaxConfig`): resolved RegInit / fence / DMA-tiling bugs and post-tiling-fix validation
4. [`IBERT_SOFTMAX.md`](IBERT_SOFTMAX.md) — hardware I-BERT Softmax (`GemminiIBertSoftmaxConfig`): stock Gemmini Softmax/Normalizer path enabled via `has_normalizations`
5. [`COMPARISON.md`](COMPARISON.md) — cross-build comparison: Softmax cycle/speedup/ulps side-by-sides, the L=128 Residual anomaly, and conclusions across scalar, PWL, and I-BERT builds
6. [`SOFTMAX_ERROR_BASELINE.md`](SOFTMAX_ERROR_BASELINE.md) — numerical error characterization vs. float gold

Also see [`sweeps/l_sweep/README.md`](sweeps/l_sweep/README.md).

Historical (legacy shared-PRNG / pre-calibration) simulator notes:

- [Spike functional baseline](baseline-tests/spike/README.md)
- [Verilator L=16 (legacy)](baseline-tests/verilator/README.md)
- [Baseline-tests index](baseline-tests/README.md)
- Pre-cal expected headers: [`correctness/expected/legacy_pre_cal/`](correctness/expected/legacy_pre_cal/)

## Making changes

- **Test-harness code** (`transformer_block_test.c`, calibration constants,
  PRNG seeding, sweep scripts) lives in `gemmini-rocc-tests`, `bareMetalC/`,
  and is yours to change freely — this is where quantization scales,
  magnitude constants, and workload shape live.
- **Gemmini's own RTL** (`generators/gemmini/src/main/scala/gemmini/*.scala`)
  and ISA/decode behavior should be treated as fixed except for two
  categories of change already established as acceptable on this project:
  (a) genuinely new hardware added as an opt-in, elaboration-gated config
  (e.g. `has_pwl_softmax`, following the `has_normalizations` pattern), and
  (b) integration wiring needed to plug a new unit into existing fence/busy
  signals. Anything else changes Gemmini's intended/default behavior and
  should be flagged explicitly before making it.
- After any change to `Gemmini`, `gemmini-rocc-tests`, or `libgemmini`, bump
  the pinning commit in the parent repo's submodule pointer (and push) so
  this repo's pinned reference stays reproducible for others.

## Background

This project explores RISC-V hardware acceleration for attention, a key
computational bottleneck in transformer inference. Built on the open-source
Gemmini/Chipyard framework and advised by Professor Tony Wu from the
Zhejiang University SPAIL Lab.
