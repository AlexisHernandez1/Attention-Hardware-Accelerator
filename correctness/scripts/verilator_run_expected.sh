#!/usr/bin/env bash
# Verilator timing/correctness run using a Spike-exported expected snapshot (no on-device gold).
# Usage: verilator_run_expected.sh <L> <D_MODEL> <D_FF> <PRNG_SEED> [TIMEOUT_CYCLES]
set -euo pipefail

if [[ $# -lt 4 ]]; then
  echo "Usage: $0 <L> <D_MODEL> <D_FF> <PRNG_SEED> [TIMEOUT_CYCLES]" >&2
  exit 2
fi

L=$1
D_MODEL=$2
D_FF=$3
SEED=$4
# Without gold, measured work is small; 20M is generous for L<=128.
TIMEOUT_CYCLES=${5:-20000000}
WALL_TIMEOUT_SECONDS=${WALL_TIMEOUT_SECONDS:-$(( 2 * 3600 ))}

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
CHIPYARD=${CHIPYARD:-/home/users/ah072084/chipyard}
ROCC="$CHIPYARD/generators/gemmini/software/gemmini-rocc-tests"
TAG="L${L}_D${D_MODEL}_F${D_FF}_seed${SEED}"
INC_DIR="$ROOT/correctness/expected/include_${TAG}"
SNAP="$ROOT/correctness/expected/${TAG}.h"
LOG_DIR="$ROOT/correctness/logs"
BIN_DIR="$ROOT/correctness/binaries"
mkdir -p "$LOG_DIR" "$BIN_DIR"

if [[ ! -f "$INC_DIR/transformer_expected.h" ]]; then
  echo "Missing $INC_DIR/transformer_expected.h — run spike_export_expected.sh first." >&2
  exit 2
fi

set +u
source "$CHIPYARD/env.sh"
set -u

CFLAGS="-DSEQ_LEN=$L -DD_MODEL=$D_MODEL -DD_FF=$D_FF -DPRNG_SEED=$SEED -DSKIP_GOLD=1 -DUSE_EXPECTED -I$INC_DIR"

echo "[verilator] Building $TAG with SKIP_GOLD + USE_EXPECTED"
make -C "$ROCC/build/bareMetalC" -B \
  -f "$ROCC/bareMetalC/Makefile" \
  abs_top_srcdir="$ROCC" src_dir="$ROCC/bareMetalC" \
  XLEN=64 PREFIX=examples-bareMetalC \
  TRANSFORMER_CFLAGS="$CFLAGS" \
  transformer_block_test-baremetal |& tee "$LOG_DIR/${TAG}.verilator_build.log"

OUT_BIN="$BIN_DIR/transformer_block_test-${TAG}-baremetal"
cp "$ROCC/build/bareMetalC/transformer_block_test-baremetal" "$OUT_BIN"

# Quick Spike check that the expected snapshot still matches (same seed, no gold).
spike --extension=gemmini "$OUT_BIN" |& tee "$LOG_DIR/${TAG}.verilator_spike_check.log"
rg -qx PASS "$LOG_DIR/${TAG}.verilator_spike_check.log"

echo "[verilator] RTL run TIMEOUT_CYCLES=$TIMEOUT_CYCLES wall=${WALL_TIMEOUT_SECONDS}s"
timeout --foreground "$WALL_TIMEOUT_SECONDS" \
  make -C "$CHIPYARD/sims/verilator" CONFIG=GemminiRocketConfig run-binary-fast \
  TIMEOUT_CYCLES="$TIMEOUT_CYCLES" \
  "BINARY=$OUT_BIN" |& tee "$LOG_DIR/${TAG}.verilator.log"

rg -qx PASS "$LOG_DIR/${TAG}.verilator.log"
echo "[verilator] PASS $TAG (snapshot $SNAP)"
