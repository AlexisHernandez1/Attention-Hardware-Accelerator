#!/usr/bin/env bash
# Official L sweep baseline: PRNG_SEED=1, independent tensor seeding.
# Spike+gold → export expected → Verilator SKIP_GOLD for L in {16,32,64,128,256}.
set -u -o pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
RUNNER="$ROOT/sweeps/run_transformer_sweep.sh"
RESULT_DIR="$ROOT/sweeps/l_sweep"
SEED=1
failed=0

for L in 16 32 64 128 256; do
  tag="L${L}_D16_F64_seed${SEED}"
  echo "========== Official L sweep: $tag =========="
  if ! "$RUNNER" "$RESULT_DIR" "$tag" "$L" 16 64 "$SEED"; then
    echo "Configuration L=$L (seed=$SEED) did not finish with Verilator PASS." >&2
    failed=1
  fi
done

python3 "$RESULT_DIR/generate_readme.py" || true
exit "$failed"
