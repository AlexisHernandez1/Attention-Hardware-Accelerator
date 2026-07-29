#!/usr/bin/env bash
# DEPRECATED: prefer ./correctness/scripts/run_attention_baseline_grid.sh
# Full-grid validation under the official calibrated baseline defaults
# (ACC_SCALE_Q/K + SCORE_DEQUANT + RMSNORM_GAIN). Checks every-stage saturation
# and gold PASS. Usage: validate_calibrated_full_grid.sh [seed_lo] [seed_hi]
set -euo pipefail

SEED_LO=${1:-1}
SEED_HI=${2:-5}
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
CHIPYARD=${CHIPYARD:-/home/users/ah072084/chipyard}
ROCC="$CHIPYARD/generators/gemmini/software/gemmini-rocc-tests"
OUT_DIR="$ROOT/correctness/magnitude_sweep"
mkdir -p "$OUT_DIR/logs"

set +u
source "$CHIPYARD/env.sh"
set -u

SUMMARY="$OUT_DIR/calibrated_full_grid_validate.tsv"
echo -e "L\tseed\tQK_sat\tany_sat\tsat_stages\tresult\tQ_abs\tK_abs\tmean_maxp" > "$SUMMARY"

any_qk=0
any_sat=0
any_fail=0
for L in 16 32 64 128 256; do
  for seed in $(seq "$SEED_LO" "$SEED_HI"); do
    TAG="val_L${L}_seed${seed}_cal"
    LOG="$OUT_DIR/logs/${TAG}.spike.log"
    CFLAGS="-DSEQ_LEN=$L -DD_MODEL=16 -DD_FF=64 -DPRNG_SEED=$seed -DDBG_RESIDUAL_RMSNORM=1"
    echo "[validate] $TAG"
    make -C "$ROCC/build/bareMetalC" -B \
      -f "$ROCC/bareMetalC/Makefile" \
      abs_top_srcdir="$ROCC" src_dir="$ROCC/bareMetalC" \
      XLEN=64 PREFIX=examples-bareMetalC \
      TRANSFORMER_CFLAGS="$CFLAGS" \
      transformer_block_test-baremetal >"$OUT_DIR/logs/${TAG}.build.log" 2>&1
    spike --extension=gemmini "$ROCC/build/bareMetalC/transformer_block_test-baremetal" |& tee "$LOG" >/dev/null

    QK_SAT=no
    if rg -q '^(Q|K) raw int8 range:.*SATURATION DETECTED' "$LOG"; then
      QK_SAT=yes; any_qk=1
    fi
    ANY_SAT=no
    STAGES=""
    if rg -q 'SATURATION DETECTED' "$LOG"; then
      ANY_SAT=yes; any_sat=1
      STAGES=$(rg -o '^([^ ]+(?: [^ ]+)*) raw int8 range:.*SATURATION DETECTED' -r '$1' "$LOG" | tr '\n' ',' | sed 's/,$//')
    fi
    RESULT=FAIL
    if rg -qx PASS "$LOG"; then RESULT=PASS; else any_fail=1; fi
    Q_ABS=$(rg -o 'Q raw int8 range:.*max_abs=([0-9]+)' -r '$1' "$LOG" | head -n1)
    K_ABS=$(rg -o 'K raw int8 range:.*max_abs=([0-9]+)' -r '$1' "$LOG" | head -n1)
    MAXP=$(rg -o 'mean_maxp=([0-9.-]+)' -r '$1' "$LOG" | head -n1)
    echo -e "${L}\t${seed}\t${QK_SAT}\t${ANY_SAT}\t${STAGES:-none}\t${RESULT}\t${Q_ABS:-0}\t${K_ABS:-0}\t${MAXP:-na}" | tee -a "$SUMMARY"
  done
done

echo
column -t -s $'\t' "$SUMMARY"
echo
echo "Q/K sat: $(awk -F'\t' 'NR>1 && $3=="yes"{c++} END{print c+0}' "$SUMMARY")/25"
echo "any-stage sat: $(awk -F'\t' 'NR>1 && $4=="yes"{c++} END{print c+0}' "$SUMMARY")/25"
echo "FAIL count: $(awk -F'\t' 'NR>1 && $6!="PASS"{c++} END{print c+0}' "$SUMMARY")"

# Grid-wide residual/RMSNorm preclip maxima after retune
python3 - "$OUT_DIR/logs" <<'PY'
import re, sys
from pathlib import Path
logdir=Path(sys.argv[1])
worst={k:(0.0,None) for k in ['res1','rms1_out','res2','rms2_out']}
for L in [16,32,64,128,256]:
  for seed in range(1,6):
    p=logdir/f'val_L{L}_seed{seed}_cal.spike.log'
    if not p.exists(): continue
    text=p.read_text()
    pres=re.findall(r'dbg residual preclip max\|raw\|=([0-9.-]+)', text)
    rms=re.findall(r'preclip max\|output_raw\|=([0-9.-]+)', text)
    if len(pres)>=2 and len(rms)>=2:
      for key,val in [('res1',float(pres[0])),('res2',float(pres[1])),
                      ('rms1_out',float(rms[0])),('rms2_out',float(rms[1]))]:
        if val>worst[key][0]:
          worst[key]=(val,(L,seed))
print('After RMSNORM_GAIN retune — grid-wide preclip maxima:')
for k,(v,loc) in worst.items():
  print(f'  {k}: {v:.3f} at L={loc[0]} seed={loc[1]}')
PY

if [[ "$any_fail" -ne 0 || "$any_sat" -ne 0 || "$any_qk" -ne 0 ]]; then
  exit 1
fi
echo "OK: 0/25 saturation across all stages; all PASS"
