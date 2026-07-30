#!/usr/bin/env bash
# Moved to correctness/archive/diagnostics/trace_all_negative_residual.sh
# (historical all_negative L=256 residual-trace investigation).
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
exec "$ROOT/correctness/archive/diagnostics/trace_all_negative_residual.sh" "$@"
