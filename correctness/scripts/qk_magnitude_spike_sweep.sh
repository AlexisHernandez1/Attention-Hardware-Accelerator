#!/usr/bin/env bash
# Analyze Q/K int8 headroom, recommend QK_WEIGHT_RAW_MAGNITUDE, and Spike-sweep
# L in {16,32,64,128,256} with that magnitude (float gold; no Verilator).
#
# Usage: qk_magnitude_spike_sweep.sh [PRNG_SEED] [D_MODEL] [D_FF] [INPUT_MAG]
set -euo pipefail

SEED=${1:-1}
D_MODEL=${2:-16}
D_FF=${3:-64}
INPUT_MAG=${4:-64}
ACC_SCALE=${ACC_SCALE:-$(python3 - <<'PY'
print(1.0/128.0)
PY
)}
HEADROOM=${HEADROOM:-100}
L_LIST=(16 32 64 128 256)

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
CHIPYARD=${CHIPYARD:-/home/users/ah072084/chipyard}
ROCC="$CHIPYARD/generators/gemmini/software/gemmini-rocc-tests"
OUT_DIR="$ROOT/correctness/magnitude_sweep"
mkdir -p "$OUT_DIR/logs"

set +u
source "$CHIPYARD/env.sh"
set -u

ANALYSIS=$(python3 - "$D_MODEL" "$INPUT_MAG" "$ACC_SCALE" "$HEADROOM" <<'PY'
import math, sys
D = int(sys.argv[1])
A = int(sys.argv[2])
acc = float(sys.argv[3])
head = float(sys.argv[4])

def stats(qk):
    sigma_raw = math.sqrt(D) * A * qk / math.sqrt(3.0)
    sigma_out = acc * sigma_raw
    return sigma_raw, sigma_out, 3.0 * sigma_out

# Recommend largest integer QK with INPUT fixed such that 3-sigma < headroom.
denom = 3.0 * acc * math.sqrt(D) * A / math.sqrt(3.0)
rec = int((head - 1e-12) / denom) if denom > 0 else 0
_, s1_cur, s3_cur = stats(56)
_, s1_rec, s3_rec = stats(rec)
print(f"D_MODEL={D}")
print(f"INPUT_RAW_MAGNITUDE={A}")
print(f"ACC_SCALE={acc}")
print(f"HEADROOM={head}")
print(f"current_QK=56 1sigma={s1_cur:.3f} 3sigma={s3_cur:.3f} exceeds127={'yes' if s3_cur > 127 else 'no'}")
print(f"recommended_QK={rec} 1sigma={s1_rec:.3f} 3sigma={s3_rec:.3f} exceeds127={'yes' if s3_rec > 127 else 'no'}")
print(rec)
PY
)

echo "$ANALYSIS" | tee "$OUT_DIR/analysis.txt"
RECOMMENDED=$(echo "$ANALYSIS" | tail -n1)
if ! [[ "$RECOMMENDED" =~ ^[0-9]+$ ]] || [[ "$RECOMMENDED" -le 0 ]]; then
  echo "Failed to compute recommended QK_WEIGHT_RAW_MAGNITUDE" >&2
  exit 1
fi

SUMMARY="$OUT_DIR/summary_seed${SEED}_qk${RECOMMENDED}.tsv"
{
  echo -e "L\tsaturation\tresult\tQ_max_abs\tK_max_abs\tmax_abs_QK"
} > "$SUMMARY"

echo
echo "=== Spike L sweep: PRNG_SEED=$SEED INPUT=$INPUT_MAG QK_W=$RECOMMENDED (gold mode) ==="

GLOBAL_MAX=0
for L in "${L_LIST[@]}"; do
  TAG="L${L}_D${D_MODEL}_F${D_FF}_seed${SEED}_qk${RECOMMENDED}"
  LOG="$OUT_DIR/logs/${TAG}.spike.log"
  CFLAGS="-DSEQ_LEN=$L -DD_MODEL=$D_MODEL -DD_FF=$D_FF -DPRNG_SEED=$SEED -DINPUT_RAW_MAGNITUDE=$INPUT_MAG -DQK_WEIGHT_RAW_MAGNITUDE=$RECOMMENDED"
  echo "[Spike] $TAG"
  make -C "$ROCC/build/bareMetalC" -B \
    -f "$ROCC/bareMetalC/Makefile" \
    abs_top_srcdir="$ROCC" src_dir="$ROCC/bareMetalC" \
    XLEN=64 PREFIX=examples-bareMetalC \
    TRANSFORMER_CFLAGS="$CFLAGS" \
    transformer_block_test-baremetal |& tee "$OUT_DIR/logs/${TAG}.build.log"
  spike --extension=gemmini "$ROCC/build/bareMetalC/transformer_block_test-baremetal" |& tee "$LOG"

  SAT=no
  if rg -q 'SATURATION DETECTED' "$LOG"; then
    SAT=yes
  fi
  RESULT=FAIL
  if rg -qx PASS "$LOG"; then
    RESULT=PASS
  fi
  Q_ABS=$(rg -o 'Q raw int8 range:.*max_abs=([0-9]+)' -r '$1' "$LOG" | head -n1)
  K_ABS=$(rg -o 'K raw int8 range:.*max_abs=([0-9]+)' -r '$1' "$LOG" | head -n1)
  Q_ABS=${Q_ABS:-0}
  K_ABS=${K_ABS:-0}
  MAX_QK=$Q_ABS
  if [[ "$K_ABS" -gt "$MAX_QK" ]]; then MAX_QK=$K_ABS; fi
  if [[ "$MAX_QK" -gt "$GLOBAL_MAX" ]]; then GLOBAL_MAX=$MAX_QK; fi
  echo -e "${L}\t${SAT}\t${RESULT}\t${Q_ABS}\t${K_ABS}\t${MAX_QK}" | tee -a "$SUMMARY"
done

echo
echo "=== Summary (QK_WEIGHT_RAW_MAGNITUDE=$RECOMMENDED, PRNG_SEED=$SEED) ==="
column -t -s $'\t' "$SUMMARY"
echo
echo "Largest observed |Q|/|K| post-scale int8 across sweep: $GLOBAL_MAX (ceiling 127)"
echo "$GLOBAL_MAX" > "$OUT_DIR/largest_qk_abs_seed${SEED}_qk${RECOMMENDED}.txt"
echo "Wrote $SUMMARY"
