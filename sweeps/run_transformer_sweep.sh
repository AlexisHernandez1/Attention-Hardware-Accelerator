#!/usr/bin/env bash
# Rebuild and run one immutable transformer_block_test configuration.
#
# Usage:
#   sweeps/run_transformer_sweep.sh <result-dir> <tag> <seq-len> <d-model> <d-ff>
#
# This script deliberately does not edit Chipyard source or Makefiles. It forces
# the existing bareMetalC target to rebuild because Make does not track changes
# to TRANSFORMER_CFLAGS as file dependencies.

set -euo pipefail

if [[ $# -ne 5 ]]; then
  echo "Usage: $0 <result-dir> <tag> <seq-len> <d-model> <d-ff>" >&2
  exit 2
fi

RESULT_DIR=$1
TAG=$2
SEQ_LEN=$3
D_MODEL=$4
D_FF=$5

CHIPYARD=${CHIPYARD:-/home/users/ah072084/chipyard}
ROCC_TESTS="$CHIPYARD/generators/gemmini/software/gemmini-rocc-tests"
BINARY="$ROCC_TESTS/build/bareMetalC/transformer_block_test-baremetal"
LOG_DIR="$RESULT_DIR/logs"
BIN_DIR="$RESULT_DIR/binaries"
DRAM_BYTES=$((0x10000000)) # 256 MiB: Rocket Chip WithDefaultMemPort.
WARNING_BYTES=$((DRAM_BYTES * 80 / 100))

mkdir -p "$LOG_DIR" "$BIN_DIR"
STATUS_LOG="$LOG_DIR/${TAG}.status.log"
: > "$STATUS_LOG"

status() {
  printf '[%s] %s\n' "$(date --iso-8601=seconds)" "$*" | tee -a "$STATUS_LOG"
}

if (( SEQ_LEN % 16 != 0 || D_MODEL % 16 != 0 || D_FF % 16 != 0 )); then
  echo "ERROR: SEQ_LEN, D_MODEL, and D_FF must all be multiples of DIM=16." >&2
  exit 2
fi

# Tensor element count shared by the quantized and float-reference pipelines:
# X, 6 weights, 13 quantized/intermediate tensors, and their float equivalents.
TENSOR_ELEMENTS=$((11 * SEQ_LEN * D_MODEL + 4 * D_MODEL * D_MODEL + \
  2 * D_MODEL * D_FF + 2 * SEQ_LEN * SEQ_LEN + SEQ_LEN * D_FF))
INT8_BYTES=$TENSOR_ELEMENTS
FLOAT_BYTES=$((4 * TENSOR_ELEMENTS))
ACC_ZERO_BYTES=$((4 * (SEQ_LEN * D_MODEL + SEQ_LEN * SEQ_LEN + SEQ_LEN * D_FF)))
STATIC_BUFFER_BYTES=$((INT8_BYTES + FLOAT_BYTES + ACC_ZERO_BYTES))

{
  echo "Configuration: L=$SEQ_LEN D_MODEL=$D_MODEL D_FF=$D_FF"
  echo "Quantized int8 tensors: $INT8_BYTES bytes"
  echo "Float32 gold-reference tensors: $FLOAT_BYTES bytes"
  echo "int32 zero-bias accumulator tensors: $ACC_ZERO_BYTES bytes"
  echo "Total declared benchmark buffers: $STATIC_BUFFER_BYTES bytes"
  echo "Verilator GemminiRocketConfig DRAM: $DRAM_BYTES bytes (256 MiB)"
  echo "Linker script: $ROCC_TESTS/riscv-tests/benchmarks/common/test.ld"
  echo "Linker placement: starts at 0x80000000; it declares no stack or DRAM upper bound."
  echo "20% free-space warning threshold: $WARNING_BYTES bytes"
} | tee "$LOG_DIR/${TAG}.footprint.log"

if (( STATIC_BUFFER_BYTES >= DRAM_BYTES )); then
  echo "ERROR: benchmark buffers exceed configured DRAM; Verilator will not run." >&2
  exit 3
fi
if (( STATIC_BUFFER_BYTES >= WARNING_BYTES )); then
  echo "WARNING: benchmark buffers use at least 80% of configured DRAM." >&2
fi

# Chipyard's activation hook probes an initially unset RISCV variable, which
# conflicts with this script's nounset mode. Restore nounset immediately after
# sourcing the environment.
set +u
source "$CHIPYARD/env.sh"
set -u

# build.sh performs the one-time configure step when build/ does not exist.
if [[ ! -d "$ROCC_TESTS/build" ]]; then
  (cd "$ROCC_TESTS" && ./build.sh bareMetalC)
fi

TRANSFORMER_CFLAGS="-DSEQ_LEN=$SEQ_LEN -DD_MODEL=$D_MODEL -DD_FF=$D_FF"
status "Build started: $TRANSFORMER_CFLAGS"
if make -C "$ROCC_TESTS/build/bareMetalC" -B \
    -f "$ROCC_TESTS/bareMetalC/Makefile" \
    abs_top_srcdir="$ROCC_TESTS" \
    src_dir="$ROCC_TESTS/bareMetalC" \
    XLEN=64 PREFIX=examples-bareMetalC \
    TRANSFORMER_CFLAGS="$TRANSFORMER_CFLAGS" \
    transformer_block_test-baremetal |& tee "$LOG_DIR/${TAG}.build.log"; then
  status "Build passed"
else
  status "Build failed"
  exit 1
fi

status "Spike pre-flight started"
if spike --extension=gemmini "$BINARY" |& tee "$LOG_DIR/${TAG}.spike.log" &&
    rg -qx 'PASS' "$LOG_DIR/${TAG}.spike.log"; then
  status "Spike pre-flight passed"
else
  status "Spike pre-flight failed; Verilator will not run"
  exit 1
fi

cp "$BINARY" "$BIN_DIR/transformer_block_test-${TAG}-baremetal"

# run-binary-fast executes the same cycle-accurate RTL but omits the enormous
# instruction-disassembly trace produced by run-binary. Test UART output, the
# stage counters, and PASS/FAIL are still captured below.
verilator_start_seconds=$(date +%s)
if [[ ${VERILATOR_TIMEOUT_SECONDS:-0} -gt 0 ]]; then
  status "Verilator started (wall-time ceiling: ${VERILATOR_TIMEOUT_SECONDS}s)"
  if timeout --foreground "$VERILATOR_TIMEOUT_SECONDS" \
      make -C "$CHIPYARD/sims/verilator" CONFIG=GemminiRocketConfig run-binary-fast \
      "BINARY=$BIN_DIR/transformer_block_test-${TAG}-baremetal" |& \
      tee "$LOG_DIR/${TAG}.verilator.log"; then
    verilator_command_passed=1
  else
    verilator_command_passed=0
  fi
else
  status "Verilator started (no wall-time ceiling)"
  if make -C "$CHIPYARD/sims/verilator" CONFIG=GemminiRocketConfig run-binary-fast \
      "BINARY=$BIN_DIR/transformer_block_test-${TAG}-baremetal" |& \
      tee "$LOG_DIR/${TAG}.verilator.log"; then
    verilator_command_passed=1
  else
    verilator_command_passed=0
  fi
fi
verilator_elapsed_seconds=$(( $(date +%s) - verilator_start_seconds ))
printf '%s\n' "$verilator_elapsed_seconds" > "$LOG_DIR/${TAG}.verilator_seconds"

if (( verilator_command_passed )) && rg -qx 'PASS' "$LOG_DIR/${TAG}.verilator.log"; then
  status "Verilator PASS (${verilator_elapsed_seconds}s)"
else
  status "Verilator FAIL or wall-time ceiling reached (${verilator_elapsed_seconds}s)"
  exit 1
fi
