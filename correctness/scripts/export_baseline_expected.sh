#!/usr/bin/env bash
# One-shot export of official baseline expected headers for L∈{16,32,64,128,256}
# at a given seed (default 1). Calibrated scales are the source defaults.
# Usage: export_baseline_expected.sh [seed]
set -euo pipefail

SEED=${1:-1}
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
SCRIPT_DIR="$ROOT/correctness/scripts"

for L in 16 32 64 128 256; do
  echo "[export-baseline] L=$L D=16 F=64 seed=$SEED"
  "$SCRIPT_DIR/spike_export_expected.sh" "$L" 16 64 "$SEED"
done
echo "Done exporting baseline expected headers for seed=$SEED"
