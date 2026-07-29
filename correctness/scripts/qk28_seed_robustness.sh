#!/usr/bin/env bash
# Multi-seed Spike saturation check at a fixed QK_WEIGHT_RAW_MAGNITUDE.
# Usage: qk28_seed_robustness.sh [QK_W] [seed_lo] [seed_hi] [D_MODEL] [D_FF]
set -euo pipefail

QK_W=${1:-28}
SEED_LO=${2:-1}
SEED_HI=${3:-5}
D_MODEL=${4:-16}
D_FF=${5:-64}
INPUT_MAG=${INPUT_MAG:-64}
L_LIST=(16 32)

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
CHIPYARD=${CHIPYARD:-/home/users/ah072084/chipyard}
ROCC="$CHIPYARD/generators/gemmini/software/gemmini-rocc-tests"
OUT_DIR="$ROOT/correctness/magnitude_sweep"
mkdir -p "$OUT_DIR/logs"

set +u
source "$CHIPYARD/env.sh"
set -u

SUMMARY="$OUT_DIR/seed_robustness_qk${QK_W}_seeds${SEED_LO}-${SEED_HI}.tsv"
echo -e "L\tseed\tsaturation\tresult\tQ_max_abs\tK_max_abs" > "$SUMMARY"

any_sat=0
any_fail=0
for L in "${L_LIST[@]}"; do
  for seed in $(seq "$SEED_LO" "$SEED_HI"); do
    TAG="L${L}_D${D_MODEL}_F${D_FF}_seed${seed}_qk${QK_W}"
    LOG="$OUT_DIR/logs/${TAG}.spike.log"
    CFLAGS="-DSEQ_LEN=$L -DD_MODEL=$D_MODEL -DD_FF=$D_FF -DPRNG_SEED=$seed -DINPUT_RAW_MAGNITUDE=$INPUT_MAG -DQK_WEIGHT_RAW_MAGNITUDE=$QK_W"
    echo "[Spike] $TAG"
    make -C "$ROCC/build/bareMetalC" -B \
      -f "$ROCC/bareMetalC/Makefile" \
      abs_top_srcdir="$ROCC" src_dir="$ROCC/bareMetalC" \
      XLEN=64 PREFIX=examples-bareMetalC \
      TRANSFORMER_CFLAGS="$CFLAGS" \
      transformer_block_test-baremetal |& tee "$OUT_DIR/logs/${TAG}.build.log" >/dev/null
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
    Q_ABS=${Q_ABS:-0}
    K_ABS=${K_ABS:-0}
    echo -e "${L}\t${seed}\t${SAT}\t${RESULT}\t${Q_ABS}\t${K_ABS}" | tee -a "$SUMMARY"
  done
done

echo
echo "=== Seed robustness summary (QK_W=$QK_W) ==="
column -t -s $'\t' "$SUMMARY"
echo
if [[ "$any_sat" -ne 0 ]]; then
  echo "WARNING: at least one seed still saturated at QK_W=$QK_W — do not adopt 28 as permanent default without further reduction."
else
  echo "No saturation across seeds ${SEED_LO}-${SEED_HI} at L in {16,32}."
fi
if [[ "$any_fail" -ne 0 ]]; then
  echo "WARNING: at least one FAIL."
  exit 1
fi
