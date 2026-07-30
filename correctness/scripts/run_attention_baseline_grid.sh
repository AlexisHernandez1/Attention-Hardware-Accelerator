#!/usr/bin/env bash
# Official single-head correctness grid (Spike gold / edge assertions).
# Default invocation runs BOTH:
#   - random PRNG baseline (EDGE_CASE_ID=0): seeds × L
#   - all 7 named edge cases (EDGE_CASE_ID=1..7): seeds × L
# Grid: seeds 1–5 × L∈{16,32,64,128,256}, D_MODEL=16, D_FF=64.
#
# Usage:
#   run_attention_baseline_grid.sh [seed_lo] [seed_hi]
#   run_attention_baseline_grid.sh --resume [seed_lo] [seed_hi]
#
# --resume skips configs that already have a spike log with an exact PASS line.
set -euo pipefail

RESUME=0
SEED_LO=1
SEED_HI=5
POSITIONAL=()
for arg in "$@"; do
  case "$arg" in
    --edge)
      echo "warning: --edge is obsolete; edge cases are part of the default grid" >&2
      ;;
    --resume) RESUME=1 ;;
    *) POSITIONAL+=("$arg") ;;
  esac
done
if [[ ${#POSITIONAL[@]} -ge 1 ]]; then
  SEED_LO=${POSITIONAL[0]}
fi
if [[ ${#POSITIONAL[@]} -ge 2 ]]; then
  SEED_HI=${POSITIONAL[1]}
fi

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
CHIPYARD=${CHIPYARD:-/home/users/ah072084/chipyard}
ROCC="$CHIPYARD/generators/gemmini/software/gemmini-rocc-tests"
OUT_DIR="$ROOT/correctness/logs/baseline_grid"
SUMMARY="$OUT_DIR/summary.tsv"
mkdir -p "$OUT_DIR"

# Always rewrite summary for a clean full-grid report. --resume only skips
# re-executing configs that already have a PASS spike log.
echo -e "case\tL\tseed\tresult" > "$SUMMARY"
if [[ "$RESUME" -eq 1 ]]; then
  echo "Resume: will skip configs with an existing PASS spike log under $OUT_DIR"
fi

set +u
source "$CHIPYARD/env.sh"
set -u

pass=0
fail=0
total=0
skipped=0

# ID 0 = random baseline; 1..7 = named edge cases (same order as transformer_block_test.c).
CASES=(
  "0:random"
  "1:all_zeros"
  "2:all_ones"
  "3:all_max_mag"
  "4:one_hot"
  "5:checkerboard"
  "6:all_negative"
  "7:near_zero"
)

already_pass() {
  local log=$1
  [[ -f "$log" ]] && grep -qx PASS "$log"
}

run_one() {
  local L=$1
  local seed=$2
  local case_id=$3
  local case_name=$4

  local TAG="${case_name}_L${L}_seed${seed}"
  local LOG="$OUT_DIR/${TAG}.spike.log"
  if [[ "$RESUME" -eq 1 ]] && already_pass "$LOG"; then
    echo "[skip] $TAG (existing PASS)"
    skipped=$((skipped + 1))
    pass=$((pass + 1))
    total=$((total + 1))
    echo -e "${case_name}\t${L}\t${seed}\tPASS" | tee -a "$SUMMARY"
    return 0
  fi

  local CFLAGS="-DSEQ_LEN=$L -DD_MODEL=16 -DD_FF=64 -DPRNG_SEED=$seed -DEDGE_CASE_ID=$case_id"
  echo "[run] $TAG"
  make -C "$ROCC/build/bareMetalC" -B \
    -f "$ROCC/bareMetalC/Makefile" \
    abs_top_srcdir="$ROCC" src_dir="$ROCC/bareMetalC" \
    XLEN=64 PREFIX=examples-bareMetalC \
    TRANSFORMER_CFLAGS="$CFLAGS" \
    transformer_block_test-baremetal >"$OUT_DIR/${TAG}.build.log" 2>&1
  set +e
  spike --extension=gemmini "$ROCC/build/bareMetalC/transformer_block_test-baremetal" \
    |& tee "$LOG" >/dev/null
  local spike_rc=${PIPESTATUS[0]}
  set -e
  total=$((total + 1))
  local RESULT=FAIL
  if grep -qx PASS "$LOG"; then
    RESULT=PASS
    pass=$((pass + 1))
  else
    fail=$((fail + 1))
    echo "  (spike_rc=$spike_rc; no exact PASS line in $LOG)" >&2
  fi
  echo -e "${case_name}\t${L}\t${seed}\t${RESULT}" | tee -a "$SUMMARY"
}

echo "Full correctness grid: cases={random + 7 edges}, seeds ${SEED_LO}-${SEED_HI}, L∈{16,32,64,128,256}"
for entry in "${CASES[@]}"; do
  CASE_ID=${entry%%:*}
  CASE_NAME=${entry#*:}
  for L in 16 32 64 128 256; do
    for seed in $(seq "$SEED_LO" "$SEED_HI"); do
      run_one "$L" "$seed" "$CASE_ID" "$CASE_NAME"
    done
  done
done

echo
column -t -s $'\t' "$SUMMARY"
echo
echo "=== per-case totals ==="
awk -F'\t' 'NR>1 {
  c[$1]++; if ($4=="PASS") p[$1]++
}
END {
  for (k in c) printf "%s\t%d/%d PASS\n", k, p[k]+0, c[k]
}' "$SUMMARY" | sort | column -t -s $'\t'
echo
echo "PASS: $pass / $total"
echo "FAIL: $fail / $total"
if [[ "$RESUME" -eq 1 ]]; then
  echo "SKIPPED (resume): $skipped"
fi
if [[ "$fail" -ne 0 ]]; then
  exit 1
fi
echo "OK: all baseline + edge-case grid configs PASS"
