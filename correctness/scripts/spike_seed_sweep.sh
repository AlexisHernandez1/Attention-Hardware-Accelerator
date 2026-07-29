#!/usr/bin/env bash
# Spike multi-seed correctness sweep with on-device float gold.
# Usage: spike_seed_sweep.sh [L] [D_MODEL] [D_FF] [seed_lo] [seed_hi]
set -euo pipefail

L=${1:-16}
D_MODEL=${2:-16}
D_FF=${3:-64}
SEED_LO=${4:-1}
SEED_HI=${5:-8}

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
CHIPYARD=${CHIPYARD:-/home/users/ah072084/chipyard}
ROCC="$CHIPYARD/generators/gemmini/software/gemmini-rocc-tests"
LOG_DIR="$ROOT/correctness/logs"
mkdir -p "$LOG_DIR"

set +u
source "$CHIPYARD/env.sh"
set -u

failed=0
for seed in $(seq "$SEED_LO" "$SEED_HI"); do
  tag="L${L}_D${D_MODEL}_F${D_FF}_seed${seed}"
  echo "=== Spike gold check $tag ==="
  make -C "$ROCC/build/bareMetalC" -B \
    -f "$ROCC/bareMetalC/Makefile" \
    abs_top_srcdir="$ROCC" src_dir="$ROCC/bareMetalC" \
    XLEN=64 PREFIX=examples-bareMetalC \
    TRANSFORMER_CFLAGS="-DSEQ_LEN=$L -DD_MODEL=$D_MODEL -DD_FF=$D_FF -DPRNG_SEED=$seed" \
    transformer_block_test-baremetal |& tee "$LOG_DIR/${tag}.build.log"

  if spike --extension=gemmini \
      "$ROCC/build/bareMetalC/transformer_block_test-baremetal" \
      |& tee "$LOG_DIR/${tag}.spike.log"; then
    if rg -qx PASS "$LOG_DIR/${tag}.spike.log"; then
      if rg -q 'SATURATION DETECTED' "$LOG_DIR/${tag}.spike.log"; then
        echo "PASS (with saturation) $tag"
      else
        echo "PASS $tag"
      fi
    else
      echo "FAIL $tag (no PASS line)"
      failed=1
    fi
  else
    echo "FAIL $tag (spike exit)"
    failed=1
  fi
done

exit "$failed"
