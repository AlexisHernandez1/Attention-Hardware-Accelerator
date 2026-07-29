#!/usr/bin/env bash
# Probe residual/RMSNorm pre-clip raw maxima (DBG_RESIDUAL_RMSNORM=1).
#
# NOTE: Residual/RMSNorm band + saturation assertions are now baked into
# transformer_block_test.c and always run. This script is optional verbose-only.
# Prefer run_attention_baseline_grid.sh for PASS/FAIL.
# Usage: probe_residual_rmsnorm.sh
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
CHIPYARD=${CHIPYARD:-/home/users/ah072084/chipyard}
ROCC="$CHIPYARD/generators/gemmini/software/gemmini-rocc-tests"
OUT_DIR="$ROOT/correctness/magnitude_sweep"
mkdir -p "$OUT_DIR/logs"

set +u
source "$CHIPYARD/env.sh"
set -u

SUMMARY="$OUT_DIR/residual_rmsnorm_probe.tsv"
echo -e "L\tseed\tres1_preclip\trms1_in\trms1_out_preclip\tres2_preclip\trms2_in\trms2_out_preclip\tany_sat\tresult" > "$SUMMARY"

for L in 16 32 64 128 256; do
  for seed in 1 2 3 4 5; do
    TAG="probe_L${L}_seed${seed}"
    LOG="$OUT_DIR/logs/${TAG}.spike.log"
    CFLAGS="-DSEQ_LEN=$L -DD_MODEL=16 -DD_FF=64 -DPRNG_SEED=$seed -DDBG_RESIDUAL_RMSNORM=1"
    echo "[probe] $TAG"
    make -C "$ROCC/build/bareMetalC" -B \
      -f "$ROCC/bareMetalC/Makefile" \
      abs_top_srcdir="$ROCC" src_dir="$ROCC/bareMetalC" \
      XLEN=64 PREFIX=examples-bareMetalC \
      TRANSFORMER_CFLAGS="$CFLAGS" \
      transformer_block_test-baremetal >"$OUT_DIR/logs/${TAG}.build.log" 2>&1
    spike --extension=gemmini "$ROCC/build/bareMetalC/transformer_block_test-baremetal" |& tee "$LOG" >/dev/null

    # Parse dbg lines in order: res1, rms1, res2, rms2
    mapfile -t PRE < <(rg -o 'dbg residual preclip max\|raw\|=([0-9.-]+)' -r '$1' "$LOG")
    mapfile -t RMSIN < <(rg -o 'dbg rmsnorm max\|input_raw\|=([0-9.-]+)' -r '$1' "$LOG")
    mapfile -t RMSOUT < <(rg -o 'preclip max\|output_raw\|=([0-9.-]+)' -r '$1' "$LOG")
    R1=${PRE[0]:-na}; R2=${PRE[1]:-na}
    I1=${RMSIN[0]:-na}; I2=${RMSIN[1]:-na}
    O1=${RMSOUT[0]:-na}; O2=${RMSOUT[1]:-na}
    SAT=no; rg -q 'SATURATION DETECTED' "$LOG" && SAT=yes
    RES=FAIL; rg -qx PASS "$LOG" && RES=PASS
    echo -e "${L}\t${seed}\t${R1}\t${I1}\t${O1}\t${R2}\t${I2}\t${O2}\t${SAT}\t${RES}" | tee -a "$SUMMARY"
  done
done

echo
python3 - "$SUMMARY" <<'PY'
import sys
from pathlib import Path
rows=[]
for line in Path(sys.argv[1]).read_text().splitlines()[1:]:
    p=line.split('\t')
    rows.append(p)
cols=['res1_preclip','rms1_in','rms1_out_preclip','res2_preclip','rms2_in','rms2_out_preclip']
idx={c:i for i,c in enumerate(['L','seed']+cols+['any_sat','result'])}
print('Grid-wide maxima (pre-clip raw levels):')
worst={}
for c in cols:
    best=None
    for r in rows:
        v=float(r[idx[c]])
        if best is None or v>best[0]:
            best=(v,int(r[0]),int(r[1]))
    worst[c]=best
    print(f'  {c}: {best[0]:.3f} at L={best[1]} seed={best[2]}')
# Recommend RMSNORM_GAIN: output_raw scales linearly with gain
# current gain=0.4; want max out preclip -> 110
max_out=max(worst['rms1_out_preclip'][0], worst['rms2_out_preclip'][0])
new_gain=0.4 * (110.0 / max_out)
print(f'\nCurrent RMSNORM_GAIN=0.4')
print(f'Worst RMSNorm preclip out raw={max_out:.3f}')
print(f'Recommended RMSNORM_GAIN={new_gain:.8f}  (maps worst→110)')
max_res=max(worst['res1_preclip'][0], worst['res2_preclip'][0])
print(f'Worst residual preclip raw={max_res:.3f} (headroom target 110; {"OK" if max_res<=110 else "NEEDS residual rescale"})')
PY
column -t -s $'\t' "$SUMMARY" | head -40
