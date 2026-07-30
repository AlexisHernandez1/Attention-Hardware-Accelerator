#!/usr/bin/env bash
# One-off diagnostic: DBG_RESIDUAL_TRACE for all_negative across L grid.
# Not part of the baseline / EDGE_CASE_SWEEP entry point.
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
CHIPYARD=${CHIPYARD:-/home/users/ah072084/chipyard}
ROCC="$CHIPYARD/generators/gemmini/software/gemmini-rocc-tests"
OUT="$ROOT/correctness/archive/diagnostics/residual_trace_all_negative"
mkdir -p "$OUT"

set +u
source "$CHIPYARD/env.sh"
set -u

run_one() {
  local L=$1
  local seed=$2
  local tag="all_negative_L${L}_seed${seed}"
  echo "[trace] $tag"
  make -C "$ROCC/build/bareMetalC" -B \
    -f "$ROCC/bareMetalC/Makefile" \
    abs_top_srcdir="$ROCC" src_dir="$ROCC/bareMetalC" \
    XLEN=64 PREFIX=examples-bareMetalC \
    TRANSFORMER_CFLAGS="-DSEQ_LEN=$L -DD_MODEL=16 -DD_FF=64 -DPRNG_SEED=$seed -DEDGE_CASE_ID=6 -DDBG_RESIDUAL_TRACE=1" \
    transformer_block_test-baremetal >"$OUT/${tag}.build.log" 2>&1
  set +e
  spike --extension=gemmini "$ROCC/build/bareMetalC/transformer_block_test-baremetal" \
    >"$OUT/${tag}.spike.log" 2>&1
  set -e
  # Extract machine-parseable block
  awk '/BEGIN_RESIDUAL_TRACE/,/END_RESIDUAL_TRACE/' "$OUT/${tag}.spike.log" \
    >"$OUT/${tag}.trace.txt"
  echo "  wrote $OUT/${tag}.trace.txt ($(wc -l < "$OUT/${tag}.trace.txt") lines)"
}

for L in 16 32 64 128 256; do
  run_one "$L" 1
done
run_one 256 4

echo "OK: traces under $OUT"
