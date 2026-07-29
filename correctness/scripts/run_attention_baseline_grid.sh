#!/usr/bin/env bash
# One-command entry point: Spike gold mode for the official single-head baseline.
# Grid: seeds 1–5 × L∈{16,32,64,128,256}, D_MODEL=16, D_FF=64.
# Calibrated ACC_SCALE_Q/K, SCORE_DEQUANT, RMSNORM_GAIN are the source defaults
# (no USE_CALIBRATED flag). Built-in residual/RMSNorm + softmax assertions always run.
# Usage: run_attention_baseline_grid.sh [seed_lo] [seed_hi]
set -euo pipefail

SEED_LO=${1:-1}
SEED_HI=${2:-5}
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
CHIPYARD=${CHIPYARD:-/home/users/ah072084/chipyard}
ROCC="$CHIPYARD/generators/gemmini/software/gemmini-rocc-tests"
OUT_DIR="$ROOT/correctness/logs/baseline_grid"
mkdir -p "$OUT_DIR"

set +u
source "$CHIPYARD/env.sh"
set -u

SUMMARY="$OUT_DIR/summary.tsv"
echo -e "L\tseed\tresult" > "$SUMMARY"

pass=0
fail=0
total=0
for L in 16 32 64 128 256; do
  for seed in $(seq "$SEED_LO" "$SEED_HI"); do
    TAG="L${L}_seed${seed}"
    LOG="$OUT_DIR/${TAG}.spike.log"
    CFLAGS="-DSEQ_LEN=$L -DD_MODEL=16 -DD_FF=64 -DPRNG_SEED=$seed"
    echo "[baseline] $TAG"
    make -C "$ROCC/build/bareMetalC" -B \
      -f "$ROCC/bareMetalC/Makefile" \
      abs_top_srcdir="$ROCC" src_dir="$ROCC/bareMetalC" \
      XLEN=64 PREFIX=examples-bareMetalC \
      TRANSFORMER_CFLAGS="$CFLAGS" \
      transformer_block_test-baremetal >"$OUT_DIR/${TAG}.build.log" 2>&1
    spike --extension=gemmini "$ROCC/build/bareMetalC/transformer_block_test-baremetal" \
      |& tee "$LOG" >/dev/null
    total=$((total + 1))
    RESULT=FAIL
    if rg -qx PASS "$LOG"; then
      RESULT=PASS
      pass=$((pass + 1))
    else
      fail=$((fail + 1))
    fi
    echo -e "${L}\t${seed}\t${RESULT}" | tee -a "$SUMMARY"
  done
done

echo
column -t -s $'\t' "$SUMMARY"
echo
echo "PASS: $pass / $total"
echo "FAIL: $fail / $total"
if [[ "$fail" -ne 0 ]]; then
  exit 1
fi
echo "OK: all baseline grid configs PASS"
