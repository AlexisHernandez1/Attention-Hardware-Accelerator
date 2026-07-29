#!/usr/bin/env bash
# Spike sweep for robust ACC_SCALE_Q/K + SCORE_DEQUANT (USE_CALIBRATED_QK_SCALES=1).
# Usage: robust_calibrated_spike_sweep.sh [seed_lo] [seed_hi]
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

SUMMARY="$OUT_DIR/robust_calibrated_sat_seeds${SEED_LO}-${SEED_HI}.tsv"
echo -e "L\tseed\tQK_sat\tany_sat\tresult\tQ_max_abs\tK_max_abs\tscores_max_abs\tmean_entropy\tmean_maxp" > "$SUMMARY"

any_qk_sat=0
any_sat=0
any_fail=0
for L in "${L_LIST[@]}"; do
  for seed in $(seq "$SEED_LO" "$SEED_HI"); do
    TAG="L${L}_D${D_MODEL}_F${D_FF}_seed${seed}_robustCal"
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

    QK_SAT=no
    if rg -q '^(Q|K) raw int8 range:.*SATURATION DETECTED' "$LOG"; then
      QK_SAT=yes
      any_qk_sat=1
    fi
    ANY_SAT=no
    if rg -q 'SATURATION DETECTED' "$LOG"; then
      ANY_SAT=yes
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
    ENT=$(rg -o 'mean_entropy=([0-9.-]+)' -r '$1' "$LOG" | head -n1)
    MAXP=$(rg -o 'mean_maxp=([0-9.-]+)' -r '$1' "$LOG" | head -n1)
    Q_ABS=${Q_ABS:-0}; K_ABS=${K_ABS:-0}; S_ABS=${S_ABS:-0}
    ENT=${ENT:-na}; MAXP=${MAXP:-na}
    echo -e "${L}\t${seed}\t${QK_SAT}\t${ANY_SAT}\t${RESULT}\t${Q_ABS}\t${K_ABS}\t${S_ABS}\t${ENT}\t${MAXP}" | tee -a "$SUMMARY"
  done
done

echo
echo "=== Robust calibrated sweep summary ==="
column -t -s $'\t' "$SUMMARY"
if [[ "$any_qk_sat" -ne 0 ]]; then
  echo "WARNING: Q/K saturation still observed."
else
  echo "OK: zero Q/K saturation across seed×L grid."
fi
if [[ "$any_sat" -ne 0 ]]; then
  echo "NOTE: some non-Q/K tensors (e.g. RMSNorm) hit int8 rails — separate from Q/K calibration."
fi
if [[ "$any_fail" -ne 0 ]]; then
  echo "WARNING: correctness FAIL observed."
  exit 1
fi
