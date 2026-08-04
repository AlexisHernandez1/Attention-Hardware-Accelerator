#!/usr/bin/env python3
"""Generate the official seed-1 L-sweep README from per-configuration run logs.

Cost note (do not change generation logic): regenerating from a full L=16…256
Verilator log set under the current counters-inline build implies an L=256 run
of roughly ~32–40 hours wall-clock (from the L=16–128 Softmax ~L² trend).
Legacy pre-counters-inline L=256 artifacts (if present) live under
legacy_pre_counters/ and are not the current baseline table.
"""

from pathlib import Path
import re

root = Path(__file__).resolve().parent
logs = root / "logs"
seed = 1
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
    tag = f"L{length}_D16_F64_seed{seed}"
    text = (logs / f"{tag}.verilator.log").read_text() if (logs / f"{tag}.verilator.log").exists() else ""
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

    footprint = (logs / f"{tag}.footprint.log").read_text() if (logs / f"{tag}.footprint.log").exists() else ""
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
    "# Official Sequence-Length Sweep Baseline (seed=1)",
    "",
    "## Setup",
    "",
    "- **Official generator:** independent per-tensor PRNG streams from `PRNG_SEED=1`.",
    "- Fixed: `D_MODEL=16`, `D_FF=64`; varied `SEQ_LEN`: 16, 32, 64, 128, 256.",
    "- Spike: float gold + export `expected_final` snapshot.",
    "- Verilator: `SKIP_GOLD=1` + `USE_EXPECTED` (exact int8 match to Spike snapshot).",
    "- Timing from cycle-accurate `GemminiRocketConfig` only; Spike cycles are not RTL performance.",
    "- Legacy shared-PRNG baselines under `baseline-tests/` are historical only — do not mix into before/after tables.",
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
        "Expected snapshots live in `correctness/expected/L*_D16_F64_seed1.h`.",
        "",
        "## Notes",
        "",
        "- Softmax share of runtime should grow with L (host scalar `expf` over L×L).",
        "- Seed 1 may still show K saturation; that is frozen for this official baseline.",
        "- A `Verilator +max-cycles timeout` means the sim-cycle budget was too small, not necessarily a functional FAIL.",
    ]
)

(root / "README.md").write_text("\n".join(lines) + "\n")
