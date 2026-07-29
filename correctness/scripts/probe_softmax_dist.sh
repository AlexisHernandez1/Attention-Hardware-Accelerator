#!/usr/bin/env bash
# Softmax fixed-vs-gold distribution probe (DBG_SOFTMAX_DIST=1).
#
# NOTE: Softmax gold-comparison assertions are now baked into
# transformer_block_test.c and always run. This script is optional verbose-only
# (per-row prints + summary TSV). Prefer run_attention_baseline_grid.sh for PASS/FAIL.
# Usage: probe_softmax_dist.sh
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
CHIPYARD=${CHIPYARD:-/home/users/ah072084/chipyard}
ROCC="$CHIPYARD/generators/gemmini/software/gemmini-rocc-tests"
OUT_DIR="$ROOT/correctness/magnitude_sweep"
mkdir -p "$OUT_DIR/logs"

set +u
source "$CHIPYARD/env.sh"
set -u

SUMMARY="$OUT_DIR/softmax_dist_vs_gold_probe.tsv"
echo -e "gain\tL\tseed\tworst_dH\tat_q\tH_fixed\tH_gold\tmax_w_fixed\tmax_w_gold\tlr_fixed\tlr_gold\tlr_gap\tworst_dmax_w\tat_q_dmax_w\tmean_dH\tmean_dmax_w\tmean_H_fixed\tmean_H_gold\tflagged_vs_gold\tflagged_diffuse\tresult" > "$SUMMARY"

for GAIN_TAG in new old; do
  if [[ "$GAIN_TAG" == "old" ]]; then
    GAIN_FLAG="-DRMSNORM_GAIN=0.4f"
    GAIN_LABEL="0.4"
  else
    GAIN_FLAG=""
    GAIN_LABEL="0.33974210"
  fi

  for L in 16 32 64 128 256; do
    for seed in 1 2 3 4 5; do
      TAG="smgold_${GAIN_TAG}_L${L}_seed${seed}"
      LOG="$OUT_DIR/logs/${TAG}.spike.log"
      CFLAGS="-DSEQ_LEN=$L -DD_MODEL=16 -DD_FF=64 -DPRNG_SEED=$seed -DDBG_SOFTMAX_DIST=1 $GAIN_FLAG"
      echo "[softmax-vs-gold] gain=$GAIN_LABEL L=$L seed=$seed"
      make -C "$ROCC/build/bareMetalC" -B \
        -f "$ROCC/bareMetalC/Makefile" \
        abs_top_srcdir="$ROCC" src_dir="$ROCC/bareMetalC" \
        XLEN=64 PREFIX=examples-bareMetalC \
        TRANSFORMER_CFLAGS="$CFLAGS" \
        transformer_block_test-baremetal >"$OUT_DIR/logs/${TAG}.build.log" 2>&1
      spike --extension=gemmini "$ROCC/build/bareMetalC/transformer_block_test-baremetal" \
        |& tee "$LOG" >/dev/null

      S=$(rg 'dbg softmax dist summary:' "$LOG" | tail -n1)
      get() { echo "$S" | rg -o "$1=([0-9.eE+-]+)" -r '$1' | head -n1; }
      RES=FAIL; rg -qx PASS "$LOG" && RES=PASS
      echo -e "${GAIN_LABEL}\t${L}\t${seed}\t$(get worst_dH)\t$(echo "$S" | rg -o 'at_q=([0-9]+)' -r '$1' | head -n1)\t$(get H_fixed)\t$(get H_gold)\t$(get max_w_fixed)\t$(get max_w_gold)\t$(get lr_fixed)\t$(get lr_gold)\t$(get lr_gap)\t$(get worst_dmax_w)\t$(echo "$S" | rg -o 'at_q_dmax_w=([0-9]+)' -r '$1' | head -n1)\t$(get mean_dH)\t$(get mean_dmax_w)\t$(get mean_H_fixed)\t$(get mean_H_gold)\t$(get flagged_uniform_vs_gold)\t$(get flagged_diffuse_vs_gold)\t${RES}" | tee -a "$SUMMARY"
    done
  done
done

echo
python3 - "$SUMMARY" <<'PY'
import sys
from pathlib import Path

rows=[]
for line in Path(sys.argv[1]).read_text().splitlines()[1:]:
    p=line.split('\t')
    rows.append(dict(
        gain=p[0], L=int(p[1]), seed=int(p[2]),
        worst_dH=float(p[3]), at_q=int(p[4]),
        H_fixed=float(p[5]), H_gold=float(p[6]),
        max_w_fixed=float(p[7]), max_w_gold=float(p[8]),
        lr_fixed=float(p[9]), lr_gold=float(p[10]), lr_gap=float(p[11]),
        worst_dmax_w=float(p[12]), at_q_dmax_w=int(p[13]),
        mean_dH=float(p[14]), mean_dmax_w=float(p[15]),
        mean_H_fixed=float(p[16]), mean_H_gold=float(p[17]),
        flag_u=int(p[18]), flag_d=int(p[19]), result=p[20],
    ))

print("=== Part 1 reminder: absolute FLAG_UNIFORM concentrated at long L ===")
print("(from prior absolute probe — flags rise with L; length dilution, not a point bug)")
print()

print("=== Fixed vs gold softmax (delta_H_eps=0.03) ===")
for gain in sorted({r['gain'] for r in rows}):
    sub=[r for r in rows if r['gain']==gain]
    worst=max(sub, key=lambda r: r['worst_dH'])
    worst_mw=min(sub, key=lambda r: r['worst_dmax_w'])
    print(f"--- RMSNORM_GAIN={gain} ---")
    print(f"{'Quantity':<42} {'Worst value':>12}  Where")
    print(f"{'worst delta_H_norm (fixed-gold)':<42} {worst['worst_dH']:12.3f}  L={worst['L']} seed={worst['seed']} layer=0 head=0 q={worst['at_q']}")
    print(f"{'  H_fixed / H_gold at that row':<42} {worst['H_fixed']:6.3f}/{worst['H_gold']:<5.3f}")
    print(f"{'  max_w_fixed / max_w_gold':<42} {worst['max_w_fixed']:6.3f}/{worst['max_w_gold']:<5.3f}")
    print(f"{'  lr_fixed / lr_gold / lr_gap':<42} {worst['lr_fixed']:6.3f}/{worst['lr_gold']:<5.3f}/{worst['lr_gap']:<.3f}")
    print(f"{'worst delta_max_w (most diffuse)':<42} {worst_mw['worst_dmax_w']:12.3f}  L={worst_mw['L']} seed={worst_mw['seed']} q={worst_mw['at_q_dmax_w']}")
    print(f"{'flagged_uniform_vs_gold (sum rows)':<42} {sum(r['flag_u'] for r in sub):12d}")
    print(f"{'flagged_diffuse_vs_gold (sum rows)':<42} {sum(r['flag_d'] for r in sub):12d}")
    print()
    print("Per-L mean_dH / flagged_vs_gold (sanity: short L ~ 0):")
    print(f"{'L':>4} {'mean_dH':>10} {'mean_dmax_w':>12} {'flags_vs_gold':>14} {'mean_H_fixed':>13} {'mean_H_gold':>12}")
    for L in [16,32,64,128,256]:
        s=[r for r in sub if r['L']==L]
        mdH=sum(r['mean_dH'] for r in s)/len(s)
        mdW=sum(r['mean_dmax_w'] for r in s)/len(s)
        flags=sum(r['flag_u'] for r in s)
        mHf=sum(r['mean_H_fixed'] for r in s)/len(s)
        mHg=sum(r['mean_H_gold'] for r in s)/len(s)
        print(f"{L:4d} {mdH:10.4f} {mdW:12.4f} {flags:14d} {mHf:13.4f} {mHg:12.4f}")
    print(f"PASS {sum(1 for r in sub if r['result']=='PASS')}/{len(sub)}")
    print()

new=[r for r in rows if r['gain']=='0.33974210']
old=[r for r in rows if r['gain']=='0.4']
diffs=0
for a,b in zip(sorted(new,key=lambda r:(r['L'],r['seed'])), sorted(old,key=lambda r:(r['L'],r['seed']))):
    if (a['worst_dH'],a['mean_dH'],a['mean_H_fixed'],a['mean_H_gold']) != (
        b['worst_dH'],b['mean_dH'],b['mean_H_fixed'],b['mean_H_gold']):
        diffs+=1
print(f"Knock-on check (softmax vs gold vs RMSNORM_GAIN): mismatches {diffs}/{len(new)}")
PY
