#!/usr/bin/env bash
# Spike L sweep (L=16,32,64,128,256) + all 8 input cases. Stops on first FAIL/timeout/build error.
# Usage: spike_l_sweep_full.sh [seed_lo] [seed_hi]
set -euo pipefail

SEED_LO=${1:-1}
SEED_HI=${2:-5}
TIMEOUT_SEC=${TIMEOUT_SEC:-600}

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
CHIPYARD=${CHIPYARD:-/home/users/ah072084/chipyard}
ROCC="$CHIPYARD/generators/gemmini/software/gemmini-rocc-tests"
OUT_DIR="$ROOT/correctness/logs/l_sweep_full"
REPORT="$OUT_DIR/report.tsv"
TRAFFIC="$OUT_DIR/traffic_detail.tsv"
mkdir -p "$OUT_DIR"

set +u
source "$CHIPYARD/env.sh"
set -u

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
LS=(16 32 64 128 256)

echo -e "L\tcase\tseed\tresult\tsaturation\tsaturated_tensors\tgold_softmax\tquant_softmax\ttotal_bytes\tx_bytes\tx_accesses\tffn_bytes" > "$REPORT"
echo -e "L\tcase\tseed\ttensor\telements\tbytes_per_access\taccesses\testimated_bytes" > "$TRAFFIC"

parse_log() {
  local log=$1
  python3 - "$log" <<'PY'
import re, sys
text = open(sys.argv[1]).read()
sat_lines = re.findall(r'^(.+?) raw int8 range:.*SATURATION DETECTED', text, re.M)
sat = "yes" if sat_lines else "no"
gold = re.search(r'Gold softmax float range: \[([^\]]+)\]', text)
quant = re.search(r'Softmax weights raw int8 range: \[([^\]]+)\]', text)
gold_s = gold.group(1) if gold else "n/a"
quant_s = quant.group(1) if quant else "n/a"
traffic = []
total = 0
for m in re.finditer(
    r'^(.+?): elements=(\d+), bytes/access=(\d+), accesses=(\d+), estimated bytes=(\d+)',
    text, re.M):
    name, elems, bpa, acc, est = m.groups()
    traffic.append((name.strip(), elems, bpa, acc, est))
    total += int(est)
x = next((t for t in traffic if t[0] == "X"), None)
x_bytes = x[4] if x else "0"
x_acc = x[3] if x else "0"
ffn = next((t for t in traffic if t[0] == "FFN intermediate"), None)
ffn_bytes = ffn[4] if ffn else "0"
print("|".join([
    sat,
    ";".join(sat_lines) if sat_lines else "-",
    gold_s,
    quant_s,
    str(total),
    str(x_bytes),
    str(x_acc),
    str(ffn_bytes),
]))
for t in traffic:
    print("TRAFFIC|" + "|".join(t))
PY
}

run_one() {
  local L=$1 seed=$2 case_id=$3 case_name=$4
  local TAG="${case_name}_L${L}_seed${seed}"
  local LOG="$OUT_DIR/${TAG}.spike.log"
  local BUILD="$OUT_DIR/${TAG}.build.log"
  local CFLAGS="-DSEQ_LEN=$L -DD_MODEL=16 -DD_FF=64 -DPRNG_SEED=$seed -DEDGE_CASE_ID=$case_id"

  echo "[run] $TAG"
  if ! make -C "$ROCC/build/bareMetalC" -B \
      -f "$ROCC/bareMetalC/Makefile" \
      abs_top_srcdir="$ROCC" src_dir="$ROCC/bareMetalC" \
      XLEN=64 PREFIX=examples-bareMetalC \
      TRANSFORMER_CFLAGS="$CFLAGS" \
      transformer_block_test-baremetal >"$BUILD" 2>&1; then
    echo -e "${L}\t${case_name}\t${seed}\tBUILD_FAIL\t-\t-\t-\t-\t-\t-\t-\t-" >> "$REPORT"
    echo "=== BUILD FAILED: $TAG ===" >&2
    tail -40 "$BUILD" >&2
    exit 2
  fi

  set +e
  timeout "$TIMEOUT_SEC" spike --extension=gemmini \
    "$ROCC/build/bareMetalC/transformer_block_test-baremetal" >"$LOG" 2>&1
  local rc=$?
  set -e

  if [[ "$rc" -eq 124 ]]; then
    echo -e "${L}\t${case_name}\t${seed}\tTIMEOUT\t-\t-\t-\t-\t-\t-\t-\t-" >> "$REPORT"
    echo "=== TIMEOUT (${TIMEOUT_SEC}s): $TAG ===" >&2
    exit 3
  fi

  local RESULT=FAIL
  grep -qx PASS "$LOG" && RESULT=PASS

  local parsed
  parsed=$(parse_log "$LOG")
  local summary rest
  summary=$(echo "$parsed" | head -1)
  rest=$(echo "$parsed" | tail -n +2)

  IFS='|' read -r sat sat_tensors gold_s quant_s total x_bytes x_acc ffn <<< "$summary"
  echo -e "${L}\t${case_name}\t${seed}\t${RESULT}\t${sat}\t${sat_tensors}\t${gold_s}\t${quant_s}\t${total}\t${x_bytes}\t${x_acc}\t${ffn}" >> "$REPORT"

  while IFS= read -r line; do
    [[ "$line" =~ ^TRAFFIC\| ]] || continue
    local tname telems tbpa tacc test
    IFS='|' read -r _ tname telems tbpa tacc test <<< "$line"
    echo -e "${L}\t${case_name}\t${seed}\t${tname}\t${telems}\t${tbpa}\t${tacc}\t${test}" >> "$TRAFFIC"
  done <<< "$rest"

  if [[ "$RESULT" != "PASS" ]]; then
    echo "=== SPIKE FAIL: $TAG (rc=$rc) ===" >&2
    rg -n "FAIL|error|SATURATION" "$LOG" | head -50 >&2 || tail -50 "$LOG" >&2
    exit 4
  fi

  echo "  OK $TAG sat=$sat total_bytes=$total gold=[$gold_s] quant=[$quant_s]"
}

echo "Spike L sweep: L∈{${LS[*]}}, 8 cases, seeds ${SEED_LO}-${SEED_HI}"
for L in "${LS[@]}"; do
  for entry in "${CASES[@]}"; do
    CASE_ID=${entry%%:*}
    CASE_NAME=${entry#*:}
    for seed in $(seq "$SEED_LO" "$SEED_HI"); do
      run_one "$L" "$seed" "$CASE_ID" "$CASE_NAME"
    done
  done
done

echo "OK: full sweep complete ($(wc -l < "$REPORT") rows incl header)"
