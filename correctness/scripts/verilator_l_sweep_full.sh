#!/usr/bin/env bash
# Cycle-accurate Verilator grid: L∈{16,32,64,128,256} × 8 cases × seeds.
# Per config: Spike dump expected → SKIP_GOLD+USE_EXPECTED rebuild → Verilator.
# Stops on FAIL or +max-cycles timeout. Saturation is flagged in the report
# (expected on some edge cases) but does not abort the grid.
#
# Usage: verilator_l_sweep_full.sh [seed_lo] [seed_hi]
#        REFERENCE_ONLY=1  → only L=16 random seed=1 (throughput probe)
set -euo pipefail

SEED_LO=${1:-1}
SEED_HI=${2:-5}
REFERENCE_ONLY=${REFERENCE_ONLY:-0}

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
CHIPYARD=${CHIPYARD:-/home/users/ah072084/chipyard}
ROCC="$CHIPYARD/generators/gemmini/software/gemmini-rocc-tests"
OUT_DIR="$ROOT/correctness/logs/verilator_l_sweep"
EXPECT_DIR="$ROOT/correctness/expected"
BIN_DIR="$ROOT/correctness/binaries/verilator_l_sweep"
REPORT="$OUT_DIR/report.tsv"
mkdir -p "$OUT_DIR" "$EXPECT_DIR" "$BIN_DIR"

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

echo -e "L\tcase\tseed\tresult\tsaturation\tsaturated_tensors\ttotal_cycles\twall_seconds\ttimeout\ttotal_bytes\tqkv\tscores\tsoftmax\tattn\tout_proj\tres1\trms1\tffn\tres2\trms2\tgemm_util" > "$REPORT"

build_one() {
  local cflags=$1
  local blog=$2
  make -C "$ROCC/build/bareMetalC" -B \
    -f "$ROCC/bareMetalC/Makefile" \
    abs_top_srcdir="$ROCC" src_dir="$ROCC/bareMetalC" \
    XLEN=64 PREFIX=examples-bareMetalC \
    TRANSFORMER_CFLAGS="$cflags" \
    transformer_block_test-baremetal >"$blog" 2>&1
}

export_expected() {
  local L=$1 seed=$2 case_id=$3 case_name=$4
  local TAG="${case_name}_L${L}_seed${seed}"
  local SNAP="$EXPECT_DIR/${TAG}.h"
  local INC="$EXPECT_DIR/include_${TAG}"
  local CFLAGS="-DSEQ_LEN=$L -DD_MODEL=16 -DD_FF=64 -DPRNG_SEED=$seed -DEDGE_CASE_ID=$case_id -DDUMP_EXPECTED=1"
  mkdir -p "$INC"
  echo "[export] $TAG"
  if ! build_one "$CFLAGS" "$OUT_DIR/${TAG}.export_build.log"; then
    echo "BUILD_FAIL export $TAG" >&2
    exit 2
  fi
  spike --extension=gemmini "$ROCC/build/bareMetalC/transformer_block_test-baremetal" \
    >"$OUT_DIR/${TAG}.export_spike.log" 2>&1
  if ! grep -qx PASS "$OUT_DIR/${TAG}.export_spike.log"; then
    echo "SPIKE_EXPORT_FAIL $TAG" >&2
    exit 4
  fi
  python3 - "$OUT_DIR/${TAG}.export_spike.log" "$SNAP" "$L" 16 64 "$seed" <<'PY'
import sys
from pathlib import Path
log_path, out_path, L, D, F, seed = sys.argv[1:7]
L, D, F, seed = map(int, (L, D, F, seed))
text = Path(log_path).read_text().splitlines()
begin = next(i for i, line in enumerate(text) if line.startswith("BEGIN_EXPECTED"))
end = next(i for i, line in enumerate(text) if line.startswith("END_EXPECTED"))
rows = [[int(x) for x in line.split()] for line in text[begin+1:end] if line.strip()]
if len(rows) != L or any(len(r) != D for r in rows):
    raise SystemExit(f"bad expected shape for {out_path}")
lines = [
    "/* Auto-generated — do not hand-edit. */",
    f"/* SEQ_LEN={L} D_MODEL={D} D_FF={F} PRNG_SEED={seed} */",
    "#ifndef TRANSFORMER_EXPECTED_H",
    "#define TRANSFORMER_EXPECTED_H",
    "#include <stdint.h>",
    f"static const int8_t expected_final[{L}][{D}] = {{",
]
for row in rows:
    lines.append("  {" + ", ".join(str(v) for v in row) + "},")
lines += ["};", "#endif", ""]
Path(out_path).write_text("\n".join(lines))
PY
  cp "$SNAP" "$INC/transformer_expected.h"
}

parse_verilator_log() {
  local log=$1
  python3 - "$log" <<'PY'
import re, sys
text = open(sys.argv[1]).read()
result = "PASS" if re.search(r'^PASS$', text, re.M) else ("TIMEOUT" if "(timeout)" in text or "max-cycles" in text.lower() else "FAIL")
sat_lines = re.findall(r'^(.+?) raw int8 range:.*SATURATION DETECTED', text, re.M)
sat = "yes" if sat_lines else "no"
total = re.search(r'^Total cycles: (\d+)', text, re.M)
stages = {}
for name in ["QKV projections","Attention scores","Softmax","Attention output",
             "Output projection","Residual add 1","RMSNorm 1","Feed-forward network",
             "Residual add 2","RMSNorm 2"]:
    # stage_names in C — check actual strings
    pass
# Actual names from stage_names[]
name_map = [
    ("QKV projections", "qkv"),
    ("Attention scores", "scores"),
    ("Softmax", "softmax"),
    ("Attention output", "attn"),
    ("Output projection", "out_proj"),
    ("Residual add 1", "res1"),
    ("RMSNorm 1", "rms1"),
    ("Feed-forward network", "ffn"),
    ("Residual add 2", "res2"),
    ("RMSNorm 2", "rms2"),
]
# Print uses stage_names — verify from source: "QKV projections" etc.
stage_vals = {}
for label, key in name_map:
    m = re.search(rf'^{re.escape(label)} cycles: (\d+) \((\d+)%\)', text, re.M)
    if m:
        stage_vals[key] = f"{m.group(1)}({m.group(2)}%)"
    else:
        stage_vals[key] = "n/a"

traffic = 0
for m in re.finditer(r'estimated bytes=(\d+)', text):
    traffic += int(m.group(1))

gemms = []
for m in re.finditer(
    r'^GEMM HW util (\S+): wall=(\d+) exe_active=(\d+) loop_matmul_active=(\d+) '
    r'ideal_mac_cycles=(\d+) exe_busy=(\d+)% pe_mesh_util=(\d+)%', text, re.M):
    gemms.append(f"{m.group(1)}:wall={m.group(2)},exe={m.group(3)},loop={m.group(4)},ideal={m.group(5)},busy={m.group(6)}%,mesh={m.group(7)}%")
gemm_s = ";".join(gemms) if gemms else "-"

print("|".join([
    result,
    sat,
    ";".join(sat_lines) if sat_lines else "-",
    total.group(1) if total else "n/a",
    str(traffic),
    stage_vals.get("qkv","n/a"),
    stage_vals.get("scores","n/a"),
    stage_vals.get("softmax","n/a"),
    stage_vals.get("attn","n/a"),
    stage_vals.get("out_proj","n/a"),
    stage_vals.get("res1","n/a"),
    stage_vals.get("rms1","n/a"),
    stage_vals.get("ffn","n/a"),
    stage_vals.get("res2","n/a"),
    stage_vals.get("rms2","n/a"),
    gemm_s,
]))
PY
}

run_one() {
  local L=$1 seed=$2 case_id=$3 case_name=$4
  local TAG="${case_name}_L${L}_seed${seed}"
  local INC="$EXPECT_DIR/include_${TAG}"
  local OUT_BIN="$BIN_DIR/transformer_block_test-${TAG}-baremetal"
  local VLOG="$OUT_DIR/${TAG}.verilator.log"
  local scale=$(( L / 16 ))
  local TIMEOUT_CYCLES=${TIMEOUT_CYCLES_OVERRIDE:-$(( 50000000 * scale * scale ))}
  # Historical L256 ~5.4h; budget 3× historical + margin.
  local WALL_TIMEOUT=${WALL_TIMEOUT_OVERRIDE:-$(( 3 * 900 * scale * scale ))}

  export_expected "$L" "$seed" "$case_id" "$case_name"

  local RTL_CFLAGS="-DSEQ_LEN=$L -DD_MODEL=16 -DD_FF=64 -DPRNG_SEED=$seed -DEDGE_CASE_ID=$case_id -DSKIP_GOLD=1 -DUSE_EXPECTED -I$INC"
  echo "[verilator-build] $TAG"
  if ! build_one "$RTL_CFLAGS" "$OUT_DIR/${TAG}.rtl_build.log"; then
    echo -e "${L}\t${case_name}\t${seed}\tBUILD_FAIL\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-" >> "$REPORT"
    echo "=== BUILD FAIL: $TAG ===" >&2
    exit 2
  fi
  cp "$ROCC/build/bareMetalC/transformer_block_test-baremetal" "$OUT_BIN"

  echo "[verilator] $TAG TIMEOUT_CYCLES=$TIMEOUT_CYCLES wall=${WALL_TIMEOUT}s"
  local t0
  t0=$(date +%s)
  set +e
  timeout --foreground "$WALL_TIMEOUT" \
    make -C "$CHIPYARD/sims/verilator" CONFIG=GemminiRocketConfig run-binary-fast \
    TIMEOUT_CYCLES="$TIMEOUT_CYCLES" \
    "BINARY=$OUT_BIN" >"$VLOG" 2>&1
  local rc=$?
  set -e
  local wall=$(( $(date +%s) - t0 ))
  echo "$wall" > "$OUT_DIR/${TAG}.verilator_seconds"

  local timed_out=no
  if [[ "$rc" -eq 124 ]] || grep -q '(timeout)' "$VLOG" || grep -qi 'max-cycles' "$VLOG"; then
    timed_out=yes
  fi

  local parsed
  parsed=$(parse_verilator_log "$VLOG")
  IFS='|' read -r result sat sat_tensors total_cyc total_bytes qkv scores softmax attn out_proj res1 rms1 ffn res2 rms2 gemm_util <<< "$parsed"

  if [[ "$timed_out" == "yes" ]]; then
    result=TIMEOUT
  fi

  echo -e "${L}\t${case_name}\t${seed}\t${result}\t${sat}\t${sat_tensors}\t${total_cyc}\t${wall}\t${timed_out}\t${total_bytes}\t${qkv}\t${scores}\t${softmax}\t${attn}\t${out_proj}\t${res1}\t${rms1}\t${ffn}\t${res2}\t${rms2}\t${gemm_util}" >> "$REPORT"

  echo "  -> $TAG result=$result wall=${wall}s cycles=$total_cyc sat=$sat timeout=$timed_out"

  if [[ "$result" == "TIMEOUT" ]]; then
    echo "=== TIMEOUT: $TAG (wall=${wall}s, +max-cycles=$TIMEOUT_CYCLES) ===" >&2
    rg -n "cycles:|Total cycles|GEMM HW|SATURATION|PASS|FAIL|timeout" "$VLOG" | head -80 >&2 || true
    exit 3
  fi
  if [[ "$result" != "PASS" ]]; then
    echo "=== FAIL: $TAG ===" >&2
    rg -n "FAIL|error|SATURATION|PASS" "$VLOG" | head -80 >&2 || tail -60 "$VLOG" >&2
    exit 4
  fi
}

if [[ "$REFERENCE_ONLY" == "1" ]]; then
  echo "Reference-only: L=16 random seed=1"
  run_one 16 1 0 random
  echo "OK: reference complete"
  exit 0
fi

echo "Verilator full grid: L∈{${LS[*]}}, 8 cases, seeds ${SEED_LO}-${SEED_HI}"
for L in "${LS[@]}"; do
  for entry in "${CASES[@]}"; do
    CASE_ID=${entry%%:*}
    CASE_NAME=${entry#*:}
    for seed in $(seq "$SEED_LO" "$SEED_HI"); do
      run_one "$L" "$seed" "$CASE_ID" "$CASE_NAME"
    done
  done
done

echo "OK: Verilator full sweep complete ($(wc -l < "$REPORT") rows incl header)"
