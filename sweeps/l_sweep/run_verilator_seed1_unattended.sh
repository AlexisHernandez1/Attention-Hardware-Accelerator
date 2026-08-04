#!/usr/bin/env bash
# Unattended Verilator L sweep: seed=1, random, L=16→32→64→128→256.
# Continues on timeout / Spike-matching saturation; hard-stops on correctness
# FAIL or unexpected saturation vs Spike.
#
# Cost note (do not change sweep logic): under the current counters-inline
# build, L=256 alone is expected to take ~32–40 hours wall-clock based on the
# L=16–128 scaling trend (Softmax ~L²). Confirm you have that budget before
# kicking off a full L=16…256 rerun. A pre-counters-inline L=256 log (unconfirmed)
# lives under legacy_pre_counters/ for reference only.
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
CHIPYARD=${CHIPYARD:-/home/users/ah072084/chipyard}
ROCC="$CHIPYARD/generators/gemmini/software/gemmini-rocc-tests"
OUT_DIR="$ROOT/sweeps/l_sweep"
SPIKE_REF="$ROOT/correctness/logs/l_sweep_full"
EXPECT_DIR="$ROOT/correctness/expected"
BIN_DIR="$ROOT/correctness/binaries/verilator_l_seed1"
MASTER_LOG="$OUT_DIR/verilator_seed1_unattended_stdout.log"
mkdir -p "$OUT_DIR/logs" "$EXPECT_DIR" "$BIN_DIR"

exec > >(tee -a "$MASTER_LOG") 2>&1

set +u
source "$CHIPYARD/env.sh"
set -u

SEED=1
CASE_ID=0
LS=(16 32 64 128 256)

# Confirm harness ceiling default (verilator_run_expected.sh).
HARNESS_DEFAULT=$(python3 - <<'PY'
import re
p="/home/users/ah072084/attention-hardware-accelerator/correctness/scripts/verilator_run_expected.sh"
text=open(p).read()
m=re.search(r'TIMEOUT_CYCLES=\$\{5:-\s*(\d+)\s*\}', text)
print(m.group(1) if m else "MISSING")
PY
)
echo "CEILING_CONFIRM harness verilator_run_expected.sh default TIMEOUT_CYCLES=$HARNESS_DEFAULT"
if [[ "$HARNESS_DEFAULT" != "50000000" ]]; then
  echo "ERROR: expected harness default 50000000, got $HARNESS_DEFAULT" >&2
  exit 2
fi
CHIPYARD_DEFAULT=$(rg -n '^TIMEOUT_CYCLES =' "$CHIPYARD/variables.mk" | head -1)
echo "CEILING_CONFIRM Chipyard variables.mk (untouched): $CHIPYARD_DEFAULT"
echo "Started: $(date --iso-8601=seconds)"

build_one() {
  local cflags=$1 blog=$2
  make -C "$ROCC/build/bareMetalC" -B \
    -f "$ROCC/bareMetalC/Makefile" \
    abs_top_srcdir="$ROCC" src_dir="$ROCC/bareMetalC" \
    XLEN=64 PREFIX=examples-bareMetalC \
    TRANSFORMER_CFLAGS="$cflags" \
    transformer_block_test-baremetal >"$blog" 2>&1
}

export_expected() {
  local L=$1
  local TAG="L${L}_D16_F64_seed${SEED}"
  local SNAP="$EXPECT_DIR/${TAG}.h"
  local INC="$EXPECT_DIR/include_${TAG}"
  mkdir -p "$INC"
  echo "[export] $TAG"
  build_one "-DSEQ_LEN=$L -DD_MODEL=16 -DD_FF=64 -DPRNG_SEED=$SEED -DEDGE_CASE_ID=$CASE_ID -DDUMP_EXPECTED=1" \
    "$OUT_DIR/logs/${TAG}.export_build.log"
  spike --extension=gemmini "$ROCC/build/bareMetalC/transformer_block_test-baremetal" \
    >"$OUT_DIR/logs/${TAG}.export_spike.log" 2>&1
  grep -qx PASS "$OUT_DIR/logs/${TAG}.export_spike.log"
  python3 - "$OUT_DIR/logs/${TAG}.export_spike.log" "$SNAP" "$L" 16 64 "$SEED" <<'PY'
import sys
from pathlib import Path
log_path, out_path, L, D, F, seed = sys.argv[1:7]
L, D, F, seed = map(int, (L, D, F, seed))
text = Path(log_path).read_text().splitlines()
begin = next(i for i, line in enumerate(text) if line.startswith("BEGIN_EXPECTED"))
end = next(i for i, line in enumerate(text) if line.startswith("END_EXPECTED"))
rows = [[int(x) for x in line.split()] for line in text[begin+1:end] if line.strip()]
assert len(rows) == L and all(len(r) == D for r in rows)
lines = [
    "/* Auto-generated — do not hand-edit. */",
    f"/* SEQ_LEN={L} D_MODEL={D} D_FF={F} PRNG_SEED={seed} random */",
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

write_report() {
  local L=$1 wall=$2 timeout_cycles=$3 vlog=$4 report=$5
  python3 - "$L" "$wall" "$timeout_cycles" "$vlog" "$report" "$SPIKE_REF" <<'PY'
import re, sys
from pathlib import Path

L, wall, ceiling, vlog, report, spike_ref = sys.argv[1:7]
L, wall, ceiling = int(L), int(wall), int(ceiling)
text = Path(vlog).read_text() if Path(vlog).exists() else ""

SW = {"Residual 1", "RMSNorm 1", "Residual 2", "Final RMSNorm"}

def sats_from(logtext):
    out = {}
    for m in re.finditer(
        r'^(.*) raw int8 range: \[(-?\d+), (-?\d+)\] max_abs=(\d+)( SATURATION DETECTED)?$',
        logtext, re.M):
        name, mn, mx, sat = m.group(1), int(m.group(2)), int(m.group(3)), bool(m.group(5))
        # Prefer banner; also apply corrected rails if banner missing
        if name in SW:
            hit = sat or (mn == -127 or mx == 127)
        else:
            hit = sat or (mn == -128 or mx == 127)
        if hit:
            out[name] = (mn, mx)
    return out

passed = bool(re.search(r'^PASS$', text, re.M))
timed_out = ('(timeout)' in text) or (re.search(r'max-cycles', text, re.I) is not None and not passed)
failed = (not passed) and (not timed_out) and (
    bool(re.search(r'^FAIL', text, re.M)) or 'FAIL:' in text or len(text) < 50)

total_m = re.search(r'^Total cycles: (\d+)', text, re.M)
total = int(total_m.group(1)) if total_m else None
pct_ceiling = (100.0 * total / ceiling) if total is not None else None

stages = [(m.group(1), int(m.group(2)), int(m.group(3)))
          for m in re.finditer(r'^(.+?) cycles: (\d+) \((\d+)%\)', text, re.M)]

traffic_total = 0
traffic_lines = []
for m in re.finditer(
    r'^(.+?): elements=(\d+), bytes/access=(\d+), accesses=(\d+), estimated bytes=(\d+)',
    text, re.M):
    traffic_lines.append(m.group(0))
    traffic_total += int(m.group(5))

gemms = re.findall(r'^GEMM HW util .+$', text, re.M)

v_sats = sats_from(text)
spike_path = Path(spike_ref) / f"random_L{L}_seed1.spike.log"
if spike_path.exists():
    s_sats = sats_from(spike_path.read_text())
    spike_note = str(spike_path)
else:
    s_sats = {}
    spike_note = f"MISSING {spike_path}"

# Compare saturation sets
unexpected = False
sat_lines = []
all_names = sorted(set(v_sats) | set(s_sats))
if not all_names:
    sat_lines.append("No saturation on any tensor (Verilator).")
    sat_lines.append(f"Spike reference ({spike_note}): no saturation." if not s_sats
                     else f"Spike reference had sat but Verilator none: {list(s_sats)} — UNEXPECTED")
    if s_sats:
        unexpected = True
else:
    for name in all_names:
        in_v = name in v_sats
        in_s = name in s_sats
        if in_v and in_s:
            sat_lines.append(
                f"{name}: Verilator sat range={v_sats[name]} ; Spike sat range={s_sats[name]} ; MATCH")
            sat_lines.append(
                f"  audit: unexpected for random baseline (Spike also had it) — FLAG if random should be sat-free")
            # Random case sat is unexpected per user expectation
            unexpected = True
        elif in_v and not in_s:
            sat_lines.append(
                f"{name}: Verilator SAT {v_sats[name]} ; Spike NO sat — MISMATCH UNEXPECTED")
            unexpected = True
        elif in_s and not in_v:
            sat_lines.append(
                f"{name}: Verilator NO sat ; Spike SAT {s_sats[name]} — MISMATCH UNEXPECTED")
            unexpected = True

# User expects little/no sat for random — any sat is noteworthy; mismatch is hard-stop
if v_sats:
    sat_lines.append(
        "NOTE: random case was expected to show little/no saturation; Verilator saw sat — see above.")

if passed:
    result = "PASS"
elif timed_out:
    result = "TIMEOUT"
else:
    result = "FAIL"

lines = []
lines.append(f"=== Verilator L={L} seed=1 case=random (GemminiRocketConfig) ===")
lines.append(f"result: {result}")
lines.append(f"wall_clock_seconds: {wall}")
lines.append(f"TIMEOUT_CYCLES (+max-cycles ceiling): {ceiling}")
lines.append(f"total_cycles: {total if total is not None else 'n/a'}")
if pct_ceiling is not None:
    lines.append(f"total_cycles_vs_ceiling: {pct_ceiling:.4f}% of +max-cycles")
else:
    lines.append("total_cycles_vs_ceiling: n/a")
lines.append(f"hit_timeout: {'yes' if timed_out else 'no'}")
lines.append("")
lines.append("--- saturation (Verilator vs Spike random_L{L}_seed1) ---".format(L=L))
lines.append(f"spike_reference: {spike_note}")
lines.extend(sat_lines)
lines.append(f"saturation_unexpected_or_mismatch: {'yes' if unexpected else 'no'}")
lines.append("")
lines.append("--- per-stage cycle breakdown ---")
if stages:
    for name, cyc, pct in stages:
        lines.append(f"{name}: {cyc} cycles ({pct}%)")
else:
    lines.append("(no stage breakdown available)")
if total is not None:
    lines.append(f"Total cycles: {total} (100%)")
lines.append("")
lines.append("--- data-movement byte estimate ---")
if traffic_lines:
    lines.extend(traffic_lines)
    lines.append(f"total_estimated_bytes: {traffic_total}")
else:
    lines.append("(no traffic estimate available)")
lines.append("")
lines.append("--- per-GEMM HW utilization (Gemmini EXE_ACTIVE_CYCLE counters; no synthesis) ---")
if gemms:
    lines.extend(gemms)
else:
    lines.append("(no GEMM HW util lines in log)")
lines.append("")
lines.append(f"raw_verilator_log: {vlog}")
lines.append("")
Path(report).write_text("\n".join(lines) + "\n")
print(f"Wrote {report} result={result} unexpected_sat={unexpected}")

# Exit: 0 continue, 4 correctness fail stop, 5 unexpected sat stop
# Timeouts always continue (capacity limit), even if sat info is incomplete.
if result == "TIMEOUT":
    sys.exit(0)
if result == "FAIL":
    sys.exit(4)
if unexpected:
    sys.exit(5)
sys.exit(0)
PY
}

for L in "${LS[@]}"; do
  TAG="L${L}_D16_F64_seed${SEED}"
  INC="$EXPECT_DIR/include_${TAG}"
  REPORT="$OUT_DIR/verilator_L${L}.log"
  VLOG="$OUT_DIR/logs/${TAG}.verilator.log"
  scale=$(( L / 16 ))
  # Harness-owned: 50M base × (L/16)^2 (same pattern as before, raised base).
  TIMEOUT_CYCLES=$(( 50000000 * scale * scale ))
  WALL_TIMEOUT=$(( 3 * 900 * scale * scale ))

  echo ""
  echo "########## L=$L TIMEOUT_CYCLES=$TIMEOUT_CYCLES WALL=${WALL_TIMEOUT}s ##########"
  if ! export_expected "$L"; then
    {
      echo "result: FAIL"
      echo "SPIKE export / expected snapshot failed for L=$L — hard stop."
    } > "$REPORT"
    echo "HARD STOP export FAIL L=$L" >&2
    exit 4
  fi

  RTL_CFLAGS="-DSEQ_LEN=$L -DD_MODEL=16 -DD_FF=64 -DPRNG_SEED=$SEED -DEDGE_CASE_ID=$CASE_ID -DSKIP_GOLD=1 -DUSE_EXPECTED -I$INC"
  echo "[rtl-build] $TAG"
  if ! build_one "$RTL_CFLAGS" "$OUT_DIR/logs/${TAG}.rtl_build.log"; then
    {
      echo "result: FAIL"
      echo "RTL binary build failed for L=$L — hard stop."
    } > "$REPORT"
    exit 4
  fi
  OUT_BIN="$BIN_DIR/transformer_block_test-${TAG}-baremetal"
  cp "$ROCC/build/bareMetalC/transformer_block_test-baremetal" "$OUT_BIN"

  echo "[verilator] $TAG ceiling=$TIMEOUT_CYCLES"
  t0=$(date +%s)
  set +e
  timeout --foreground "$WALL_TIMEOUT" \
    make -C "$CHIPYARD/sims/verilator" CONFIG=GemminiRocketConfig run-binary-fast \
    TIMEOUT_CYCLES="$TIMEOUT_CYCLES" \
    "BINARY=$OUT_BIN" >"$VLOG" 2>&1
  rc=$?
  set -e
  wall=$(( $(date +%s) - t0 ))
  echo "$wall" > "$OUT_DIR/logs/${TAG}.verilator_seconds"
  # Confirm +max-cycles in the make invocation echoed into the log
  rg -n "max-cycles=" "$VLOG" | head -3 || true

  set +e
  write_report "$L" "$wall" "$TIMEOUT_CYCLES" "$VLOG" "$REPORT"
  wr=$?
  set -e
  echo "L=$L done wall=${wall}s write_rc=$wr make_rc=$rc"

  if [[ "$wr" -eq 4 ]]; then
    echo "HARD STOP: correctness FAIL at L=$L" >&2
    exit 4
  fi
  if [[ "$wr" -eq 5 ]]; then
    echo "HARD STOP: unexpected/mismatched saturation at L=$L" >&2
    exit 5
  fi
done

echo "Finished all L values: $(date --iso-8601=seconds)"
echo "Reports: $OUT_DIR/verilator_L{16,32,64,128,256}.log"
