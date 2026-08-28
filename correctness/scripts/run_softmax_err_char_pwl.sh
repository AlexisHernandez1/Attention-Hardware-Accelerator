#!/usr/bin/env bash
# Softmax PWL (USE_PWL_SOFTMAX=1) error characterization sweep (Spike, random only).
# Requires CHAR_SOFTMAX_ERR instrumentation in transformer_block_test.c.
#
# Usage:
#   ./run_softmax_err_char_pwl.sh
# Env:
#   CHIPYARD  (default: /home/users/ah072084/chipyard)
#   TIMEOUT_SEC (default: 1800)
set -euo pipefail

CHIPYARD=${CHIPYARD:-/home/users/ah072084/chipyard}
ROCC="$CHIPYARD/generators/gemmini/software/gemmini-rocc-tests"
OUT_DIR="$CHIPYARD/tmp_baseline_validation/softmax_err_char_pwl"
LOG_DIR="$OUT_DIR/logs"
REPORT="$OUT_DIR/report.tsv"
SUMMARY="$OUT_DIR/SUMMARY.txt"
TIMEOUT_SEC=${TIMEOUT_SEC:-1800}
LS=(16 32 64 128 256)
SEEDS=(1 2 3 4 5)

mkdir -p "$LOG_DIR"
set +u
source "$CHIPYARD/env.sh"
set -u

echo -e "L\tseed\tresult\tsoft_mae\tsoft_max\tsoft_max_ulps\tsoft_mae_ulps\tfinal_mae\tfinal_max\tfinal_max_over_tol\tn_over_tol" > "$REPORT"

run_one() {
  local L=$1 seed=$2
  local TAG="random_L${L}_seed${seed}"
  local CFLAGS="-DSEQ_LEN=$L -DD_MODEL=16 -DD_FF=64 -DPRNG_SEED=$seed -DEDGE_CASE_ID=0 -DCHAR_SOFTMAX_ERR=1 -DUSE_PWL_SOFTMAX=1"
  local BUILD="$LOG_DIR/${TAG}.build.log"
  local LOG="$LOG_DIR/${TAG}.spike.log"

  echo "[char] $TAG"
  if ! make -C "$ROCC/build/bareMetalC" -B \
      -f "$ROCC/bareMetalC/Makefile" \
      abs_top_srcdir="$ROCC" src_dir="$ROCC/bareMetalC" \
      XLEN=64 PREFIX=examples-bareMetalC \
      TRANSFORMER_CFLAGS="$CFLAGS" \
      transformer_block_test-baremetal >"$BUILD" 2>&1; then
    echo -e "${L}\t${seed}\tBUILD_FAIL\t\t\t\t\t\t\t\t" >> "$REPORT"
    echo "BUILD_FAIL $TAG" >&2
    tail -20 "$BUILD" >&2
    return 2
  fi

  set +e
  timeout "$TIMEOUT_SEC" spike --extension=gemmini \
    "$ROCC/build/bareMetalC/transformer_block_test-baremetal" >"$LOG" 2>&1
  local rc=$?
  set -e
  if [[ "$rc" -eq 124 ]]; then
    echo -e "${L}\t${seed}\tTIMEOUT\t\t\t\t\t\t\t\t" >> "$REPORT"
    return 3
  fi

  local result=FAIL
  if grep -qx PASS "$LOG"; then
    result=PASS
  fi

  # Strict single-pass parse: one regex per line, matching print_float3 (%d.%03d).
  # No generic grab()+overwrite — format drift fails loudly.
  python3 - "$L" "$seed" "$result" "$LOG" "$REPORT" <<'PY'
import re, sys
from pathlib import Path

L, seed, result, log_path, report_path = sys.argv[1:6]
text = Path(log_path).read_text()

# print_float3 => optional '-', digits, '.', exactly 3 fractional digits
F = r'-?\d+\.\d{3}'

soft_re = re.compile(
    rf'(?m)^SOFTMAX_QUANT_ERR L=(?P<L>\d+) seed=(?P<seed>\d+) n=(?P<n>\d+) '
    rf'mae=(?P<mae>{F}) max=(?P<max>{F}) max_ulps=(?P<max_ulps>{F}) '
    rf'mae_ulps=(?P<mae_ulps>{F}) quant_scale=(?P<quant_scale>{F}) '
    rf'hist_ulps=\[0-0\.5\)=(?P<h0>\d+) \[0\.5-1\)=(?P<h1>\d+) \[1-2\)=(?P<h2>\d+) '
    rf'\[2-4\)=(?P<h3>\d+) \[4-8\)=(?P<h4>\d+) \[8-16\)=(?P<h5>\d+) '
    rf'\[16-32\)=(?P<h6>\d+) \[32\+\)=(?P<h7>\d+)'
)
fin_re = re.compile(
    rf'(?m)^FINAL_GOLD_ERR L=(?P<L>\d+) seed=(?P<seed>\d+) n=(?P<n>\d+) '
    rf'mae=(?P<mae>{F}) max=(?P<max>{F}) tol=(?P<tol>{F}) '
    rf'max_over_tol=(?P<max_over_tol>{F}) n_over_tol=(?P<n_over_tol>\d+) '
    rf'hist_tol=\[0-0\.25\)=(?P<t0>\d+) \[0\.25-0\.5\)=(?P<t1>\d+) '
    rf'\[0\.5-0\.75\)=(?P<t2>\d+) \[0\.75-1\)=(?P<t3>\d+) '
    rf'\[1-2\)=(?P<t4>\d+) \[2-4\)=(?P<t5>\d+) \[4-8\)=(?P<t6>\d+) \[8\+\)=(?P<t7>\d+)'
)

sm = soft_re.search(text)
if sm is None:
    raise SystemExit(
        f"PARSE_FAIL {log_path}: no SOFTMAX_QUANT_ERR line matching expected format "
        f"(need mae=/max=/max_ulps=/mae_ulps=/hist_ulps=...)"
    )
# hist_ulps must be present and non-empty (all eight bins captured as digits)
hist_vals = [sm.group(f'h{i}') for i in range(8)]
if any(v is None or v == '' for v in hist_vals):
    raise SystemExit(f"PARSE_FAIL {log_path}: hist_ulps bins missing/empty on SOFTMAX_QUANT_ERR")

fm = fin_re.search(text)
if fm is None:
    raise SystemExit(
        f"PARSE_FAIL {log_path}: no FINAL_GOLD_ERR line matching expected format"
    )

# Cross-check L/seed embedded in the log against the sweep keys
if sm.group('L') != L or sm.group('seed') != seed:
    raise SystemExit(
        f"PARSE_FAIL {log_path}: SOFTMAX_QUANT_ERR L/seed "
        f"{sm.group('L')}/{sm.group('seed')} != sweep {L}/{seed}"
    )
if fm.group('L') != L or fm.group('seed') != seed:
    raise SystemExit(
        f"PARSE_FAIL {log_path}: FINAL_GOLD_ERR L/seed "
        f"{fm.group('L')}/{fm.group('seed')} != sweep {L}/{seed}"
    )

row = '\t'.join([
    L, seed, result,
    sm.group('mae'), sm.group('max'), sm.group('max_ulps'), sm.group('mae_ulps'),
    fm.group('mae'), fm.group('max'), fm.group('max_over_tol'), fm.group('n_over_tol'),
])
with open(report_path, 'a') as f:
    f.write(row + '\n')
print(row)
PY
}

fail=0
for L in "${LS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    if ! run_one "$L" "$seed"; then
      fail=1
    fi
  done
done

# Aggregate histograms with explicit per-log match checks (never silent no-op).
python3 - "$LOG_DIR" "$SUMMARY" "$REPORT" <<'PY'
import re, sys, csv
from pathlib import Path
from collections import defaultdict

log_dir = Path(sys.argv[1])
summary_path = Path(sys.argv[2])
report_path = Path(sys.argv[3])

F = r'-?\d+\.\d{3}'
hist_re = re.compile(
    rf'(?m)^SOFTMAX_QUANT_ERR .* '
    rf'hist_ulps=\[0-0\.5\)=(\d+) \[0\.5-1\)=(\d+) \[1-2\)=(\d+) \[2-4\)=(\d+) '
    rf'\[4-8\)=(\d+) \[8-16\)=(\d+) \[16-32\)=(\d+) \[32\+\)=(\d+)'
)
keys = ['[0-0.5)', '[0.5-1)', '[1-2)', '[2-4)', '[4-8)', '[8-16)', '[16-32)', '[32+)']
bins = defaultdict(int)
parsed = 0
missing = []
expected = 25

logs = sorted(log_dir.glob('random_L*_seed*.spike.log'))
for log in logs:
    text = log.read_text()
    m = hist_re.search(text)
    if m is None:
        print(f"WARNING: hist_ulps not found or malformed in {log.name} — skipping histogram contrib",
              file=sys.stderr)
        missing.append(log.name)
        continue
    for k, v in zip(keys, m.groups()):
        bins[k] += int(v)
    parsed += 1

lines = []
lines.append("Softmax PWL quant-error characterization summary (USE_PWL_SOFTMAX=1)")
lines.append(f"spike logs found: {len(logs)}")
lines.append(f"histogram parse OK: {parsed}/{expected} (expected {expected} configs)")
if missing:
    lines.append(f"histogram parse MISSING/MALFORMED ({len(missing)}): {', '.join(missing)}")
else:
    lines.append("histogram parse MISSING/MALFORMED: none")

total = sum(bins.values())
lines.append("")
lines.append("Pooled softmax |err| hist (ulps of QUANT_SCALE) across parsed logs:")
if total == 0:
    lines.append("  ERROR: zero histogram elements aggregated — refusing empty 'no error' result")
    print('\n'.join(lines))
    summary_path.write_text('\n'.join(lines) + '\n')
    raise SystemExit("AGGREGATE_FAIL: hist total==0 (no successful hist_ulps parses)")
for k in keys:
    lines.append(f"  {k}: {bins[k]} ({100.0 * bins[k] / total:.2f}%)")
lines.append(f"  total elems: {total}")

# Report-level aggregates (only rows with numeric soft_mae)
rows = list(csv.DictReader(report_path.open(), delimiter='\t'))
ok = [r for r in rows if r.get('result') == 'PASS' and r.get('soft_mae')]
if ok:
    def fget(r, k):
        return float(r[k])
    soft_maes = [fget(r, 'soft_mae') for r in ok]
    soft_maxs = [fget(r, 'soft_max') for r in ok]
    soft_ulps = [fget(r, 'soft_max_ulps') for r in ok]
    fin_maxs = [fget(r, 'final_max') for r in ok]
    fin_ots = [fget(r, 'final_max_over_tol') for r in ok]
    lines.append("")
    lines.append(f"PASS rows with parsed metrics: {len(ok)}/{len(rows)}")
    lines.append(
        f"soft_mae range [{min(soft_maes):.6f}, {max(soft_maes):.6f}] "
        f"mean={sum(soft_maes)/len(soft_maes):.6f}"
    )
    lines.append(f"soft_max range [{min(soft_maxs):.6f}, {max(soft_maxs):.6f}]")
    lines.append(f"soft_max_ulps range [{min(soft_ulps):.3f}, {max(soft_ulps):.3f}]")
    lines.append(f"final_max range [{min(fin_maxs):.6f}, {max(fin_maxs):.6f}]")
    lines.append(
        f"final_max_over_tol range [{min(fin_ots):.4f}, {max(fin_ots):.4f}] "
        f"(tol = 5×QUANT_SCALE = 5/128)"
    )
    over = [r for r in ok if int(float(r['n_over_tol'])) > 0]
    lines.append(f"configs with n_over_tol>0: {len(over)}")

text = '\n'.join(lines)
print(text)
summary_path.write_text(text + '\n')

if parsed != expected:
    raise SystemExit(
        f"AGGREGATE_FAIL: histogram parse {parsed}/{expected} — not all configs contributed"
    )
PY

echo "OK report=$REPORT summary=$SUMMARY"
exit "$fail"
