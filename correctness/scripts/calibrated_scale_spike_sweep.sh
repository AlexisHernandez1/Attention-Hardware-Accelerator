#!/usr/bin/env bash
# Evaluate calibrated ACC_SCALE_Q/K (QK_W=56) on Spike across seeds/L.
# Usage: calibrated_scale_spike_sweep.sh [seed_lo] [seed_hi]
set -euo pipefail

SEED_LO=${1:-1}
SEED_HI=${2:-5}
D_MODEL=16
D_FF=64
L_LIST=(16 32 64 128 256)

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
CHIPYARD=${CHIPYARD:-/home/users/ah072084/chipyard}
ROCC="$CHIPYARD/generators/gemmini/software/gemmini-rocc-tests"
OUT_DIR="$ROOT/correctness/magnitude_sweep"
mkdir -p "$OUT_DIR/logs"

set +u
source "$CHIPYARD/env.sh"
set -u

SUMMARY="$OUT_DIR/calibrated_scale_sat_seeds${SEED_LO}-${SEED_HI}.tsv"
echo -e "L\tseed\tsaturation\tresult\tQ_max_abs\tK_max_abs\tscores_max_abs" > "$SUMMARY"

any_sat=0
any_fail=0
for L in "${L_LIST[@]}"; do
  for seed in $(seq "$SEED_LO" "$SEED_HI"); do
    TAG="L${L}_D${D_MODEL}_F${D_FF}_seed${seed}_calQK"
    LOG="$OUT_DIR/logs/${TAG}.spike.log"
    CFLAGS="-DSEQ_LEN=$L -DD_MODEL=$D_MODEL -DD_FF=$D_FF -DPRNG_SEED=$seed -DQK_WEIGHT_RAW_MAGNITUDE=56 -DUSE_CALIBRATED_QK_SCALES=1"
    echo "[Spike] $TAG"
    make -C "$ROCC/build/bareMetalC" -B \
      -f "$ROCC/bareMetalC/Makefile" \
      abs_top_srcdir="$ROCC" src_dir="$ROCC/bareMetalC" \
      XLEN=64 PREFIX=examples-bareMetalC \
      TRANSFORMER_CFLAGS="$CFLAGS" \
      transformer_block_test-baremetal >"$OUT_DIR/logs/${TAG}.build.log" 2>&1
    spike --extension=gemmini "$ROCC/build/bareMetalC/transformer_block_test-baremetal" |& tee "$LOG" >/dev/null

    SAT=no
    if rg -q 'SATURATION DETECTED' "$LOG"; then
      SAT=yes
      any_sat=1
    fi
    RESULT=FAIL
    if rg -qx PASS "$LOG"; then
      RESULT=PASS
    else
      any_fail=1
    fi
    Q_ABS=$(rg -o 'Q raw int8 range:.*max_abs=([0-9]+)' -r '$1' "$LOG" | head -n1)
    K_ABS=$(rg -o 'K raw int8 range:.*max_abs=([0-9]+)' -r '$1' "$LOG" | head -n1)
    S_ABS=$(rg -o 'Attention scores raw int8 range:.*max_abs=([0-9]+)' -r '$1' "$LOG" | head -n1)
    Q_ABS=${Q_ABS:-0}; K_ABS=${K_ABS:-0}; S_ABS=${S_ABS:-0}
    echo -e "${L}\t${seed}\t${SAT}\t${RESULT}\t${Q_ABS}\t${K_ABS}\t${S_ABS}" | tee -a "$SUMMARY"
  done
done

echo
echo "=== Calibrated ACC_SCALE_Q/K saturation summary ==="
column -t -s $'\t' "$SUMMARY"
if [[ "$any_sat" -ne 0 ]]; then
  echo "WARNING: saturation still observed under calibrated scales."
fi
if [[ "$any_fail" -ne 0 ]]; then
  echo "WARNING: correctness FAIL observed."
  exit 1
fi
