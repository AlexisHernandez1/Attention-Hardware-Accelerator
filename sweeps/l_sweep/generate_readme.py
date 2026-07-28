#!/usr/bin/env python3
"""Generate the L-sweep README from immutable per-configuration run logs."""

from pathlib import Path
import re

root = Path(__file__).resolve().parent
logs = root / "logs"
stages = [
    "QKV projections",
    "Attention scores",
    "Softmax",
    "Attention output",
    "Output projection",
    "Residual add 1",
    "RMSNorm 1",
    "Feed-forward network",
    "Residual add 2",
    "RMSNorm 2",
]


def match(pattern: str, text: str, default: str = "—") -> str:
    found = re.search(pattern, text, re.MULTILINE)
    return found.group(1) if found else default


rows = []
footprints = []
for length in (16, 32, 64, 128, 256):
    tag = f"L{length}_D16_F64"
    verilator_log = (logs / f"{tag}.verilator.log")
    text = verilator_log.read_text() if verilator_log.exists() else ""
    values = []
    for stage in stages:
        item = re.search(
            rf"^{re.escape(stage)} cycles: (\d+) \((\d+)%\)$",
            text,
            re.MULTILINE,
        )
        values.append(f"{item.group(1)} ({item.group(2)}%)" if item else "—")
    total = match(r"^Total cycles: (\d+) \(100%\)$", text)
    if re.search(r"^PASS$", text, re.MULTILINE):
        result = "PASS"
    elif "*** FAILED ***" in text and "(timeout)" in text:
        result = "Verilator +max-cycles timeout"
    else:
        result = "FAIL / incomplete"
    rows.append([str(length), total, *values, result])

    footprint_log = logs / f"{tag}.footprint.log"
    footprint = footprint_log.read_text() if footprint_log.exists() else ""
    footprints.append(
        [
            str(length),
            match(r"Quantized int8 tensors: (\d+ bytes)", footprint),
            match(r"Float32 gold-reference tensors: (\d+ bytes)", footprint),
            match(r"int32 zero-bias accumulator tensors: (\d+ bytes)", footprint),
            match(r"Total declared benchmark buffers: (\d+ bytes)", footprint),
        ]
    )

headers = ["L", "Total cycles", *stages, "Result"]
lines = [
    "# Sequence-Length Sweep Baseline",
    "",
    "## Setup",
    "",
    "- Fixed dimensions: `D_MODEL=16`, `D_FF=64`; varied `SEQ_LEN`: 16, 32, 64, 128, 256.",
    "- Every point is force-rebuilt with `TRANSFORMER_CFLAGS`; each first passes Spike before RTL simulation.",
    "- Timing numbers come only from cycle-accurate Verilator using `GemminiRocketConfig`.",
    "- Spike is a functional pre-flight check; its reported cycles are not hardware-performance data.",
    "- Each Verilator log contains timestamped build, Spike, and Verilator status lines.",
    "",
    "## Verilator Results",
    "",
    "Each stage cell is `cycles (percent of total)`.",
    "",
    "| " + " | ".join(headers) + " |",
    "| " + " | ".join(["---"] * len(headers)) + " |",
]
lines.extend("| " + " | ".join(row) + " |" for row in rows)
lines.extend(
    [
        "",
        "## Memory-Footprint Checks",
        "",
        "| L | int8 tensors | float32 gold tensors | int32 zero-bias tensors | total declared buffers |",
        "| --- | --- | --- | --- | --- |",
    ]
)
lines.extend("| " + " | ".join(row) + " |" for row in footprints)
lines.extend(
    [
        "",
        "The bare-metal linker script places the image at `0x80000000` but does not declare a stack or DRAM upper bound. "
        "`GemminiRocketConfig` uses Rocket Chip's default 256 MiB external-memory window; the table reports the benchmark's declared tensors only.",
        "",
        "## Sanity Check",
        "",
        "The `L=16` result is expected to match the previously validated manual baseline: `145744` total cycles and `PASS`. "
        "A mismatch indicates that the sweep pipeline or simulator configuration needs investigation before comparing larger points.",
        "",
        "## Incomplete Points",
        "",
        "A `Verilator +max-cycles timeout` result means the simulator's fixed `+max-cycles=10000000` limit expired before the benchmark printed `PASS` or `FAIL`. "
        "It is distinct from the script's wall-time ceiling and does not establish numerical correctness or timing for that point.",
    ]
)

(root / "README.md").write_text("\n".join(lines) + "\n")
