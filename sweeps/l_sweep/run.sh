#!/usr/bin/env bash
# L sweep: hold model width constant while increasing sequence length.
set -u -o pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
RUNNER="$ROOT/sweeps/run_transformer_sweep.sh"
RESULT_DIR="$ROOT/sweeps/l_sweep"
failed=0

if ! "$RUNNER" "$RESULT_DIR" "L16_D16_F64" 16 16 64; then
  echo "Configuration L=16 failed; later L points will not run." >&2
  exit 1
fi

baseline_seconds=$(<"$RESULT_DIR/logs/L16_D16_F64.verilator_seconds")
echo "L=16 Verilator wall time: ${baseline_seconds}s"

for L in 32 64 128 256; do
  # Scalar softmax and attention-score storage grow with L^2. Scale the
  # generous 2x-L=16 ceiling by that work factor so healthy larger points are
  # not falsely classified as hung.
  scale=$(( (L / 16) * (L / 16) ))
  timeout_seconds=$(( 2 * baseline_seconds * scale ))
  if ! VERILATOR_TIMEOUT_SECONDS="$timeout_seconds" \
      "$RUNNER" "$RESULT_DIR" "L${L}_D16_F64" "$L" 16 64; then
    echo "Configuration L=$L did not finish with Verilator PASS." >&2
    failed=1
  fi
done

exit "$failed"
