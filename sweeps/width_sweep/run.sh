#!/usr/bin/env bash
# Width sweep: hold sequence length constant while scaling transformer width.
set -u -o pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
RUNNER="$ROOT/sweeps/run_transformer_sweep.sh"
RESULT_DIR="$ROOT/sweeps/width_sweep"
failed=0

for D_MODEL in 16 32 64 128 256; do
  D_FF=$((4 * D_MODEL))
  if ! "$RUNNER" "$RESULT_DIR" "L16_D${D_MODEL}_F${D_FF}" 16 "$D_MODEL" "$D_FF"; then
    echo "Configuration D_MODEL=$D_MODEL D_FF=$D_FF failed; its Verilator step was not run." >&2
    failed=1
  fi
done

exit "$failed"
