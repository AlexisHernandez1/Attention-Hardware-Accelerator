#!/usr/bin/env python3
"""Side-by-side Q/K/score/softmax metrics for three scale strategies."""
from __future__ import annotations

import math
import importlib.util
from pathlib import Path

import numpy as np

SPEC = importlib.util.spec_from_file_location(
    "div",
    str(Path(__file__).resolve().parent / "diversity_check.py"),
)
div = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(div)

QUANT = 1.0 / 128.0
TARGET = 105.0


def raw_mm(a, b, transpose_b=False):
    bb = b.T if transpose_b else b
    return (a.astype(np.int32) @ bb.astype(np.int32)).astype(np.float64)


def scale_clip(acc, scale):
    scaled = acc * scale
    rounded = np.where(scaled >= 0, np.floor(scaled + 0.5), np.ceil(scaled - 0.5))
    return np.clip(rounded, -128, 127).astype(np.int16)


def summarize(label, Q, K, scores, L):
    Sf = scores.astype(np.float64) * QUANT
    sm = div.softmax_rows(Sf)
    H = div.row_entropy(sm)
    lnL = math.log(L)
    q_sat = bool(np.any((Q <= -128) | (Q >= 127)))
    k_sat = bool(np.any((K <= -128) | (K >= 127)))
    s_sat = bool(np.any((scores <= -128) | (scores >= 127)))
    return {
        "label": label,
        "Q_max_abs": int(np.max(np.abs(Q))),
        "K_max_abs": int(np.max(np.abs(K))),
        "Q_sat": q_sat,
        "K_sat": k_sat,
        "score_raw": (int(scores.min()), int(scores.max())),
        "score_std": float(Sf.std()),
        "score_range": float(Sf.max() - Sf.min()),
        "H_mean": float(H.mean()),
        "dH": float(lnL - H.mean()),
        "maxp_mean": float(sm.max(axis=1).mean()),
        "uniform_maxp": 1.0 / L,
        "score_sat": s_sat,
    }


def mode_qk56_quant(seed, L, D, D_FF):
    t = div.tensors_mix(seed, L, D, D_FF, 56)
    Q = scale_clip(raw_mm(t["X"], t["W_q"]), QUANT)
    K = scale_clip(raw_mm(t["X"], t["W_k"]), QUANT)
    S = scale_clip(raw_mm(Q, K, True), QUANT / math.sqrt(D))
    return summarize("A: QK=56 + ACC=QUANT (old default)", Q, K, S, L)


def mode_qk28_quant(seed, L, D, D_FF):
    t = div.tensors_mix(seed, L, D, D_FF, 28)
    Q = scale_clip(raw_mm(t["X"], t["W_q"]), QUANT)
    K = scale_clip(raw_mm(t["X"], t["W_k"]), QUANT)
    S = scale_clip(raw_mm(Q, K, True), QUANT / math.sqrt(D))
    return summarize("B: QK=28 + ACC=QUANT (anti-sat shrink)", Q, K, S, L)


def mode_calibrated(seed, L, D, D_FF, acc_q=None, acc_k=None):
    t = div.tensors_mix(seed, L, D, D_FF, 56)
    acc_q_raw = raw_mm(t["X"], t["W_q"])
    acc_k_raw = raw_mm(t["X"], t["W_k"])
    if acc_q is None:
        acc_q = TARGET / (3.0 * acc_q_raw.std())
    if acc_k is None:
        acc_k = TARGET / (3.0 * acc_k_raw.std())
    Q = scale_clip(acc_q_raw, acc_q)
    K = scale_clip(acc_k_raw, acc_k)
    S = scale_clip(raw_mm(Q, K, True), QUANT / math.sqrt(D))  # existing SCORE_SCALE
    row = summarize(
        f"C: QK=56 + ACC_Q/K calibrated (3σ→{TARGET:.0f}), SCORE=QUANT/√D",
        Q, K, S, L,
    )
    row["ACC_SCALE_Q"] = float(acc_q)
    row["ACC_SCALE_K"] = float(acc_k)
    row["raw_std_Q"] = float(acc_q_raw.std())
    row["raw_std_K"] = float(acc_k_raw.std())
    return row


def print_row(r):
    print(r["label"])
    print(
        f"  Q max_abs={r['Q_max_abs']} sat={r['Q_sat']} | "
        f"K max_abs={r['K_max_abs']} sat={r['K_sat']}"
    )
    print(
        f"  scores raw={r['score_raw']} sat={r['score_sat']} | "
        f"std={r['score_std']:.4f} range={r['score_range']:.4f}"
    )
    print(
        f"  softmax H={r['H_mean']:.4f} (ln L={math.log(32):.4f}, dH={r['dH']:.4f}) | "
        f"mean maxp={r['maxp_mean']:.4f} vs 1/L={r['uniform_maxp']:.4f}"
    )


def main():
    seed, L, D, D_FF = 1, 32, 16, 64
    # Frozen constants matching transformer_block_test.c
    ACC_Q = 0.00836620
    ACC_K = 0.00656261
    print(f"Comparison at seed={seed} L={L} D={D}")
    print()
    rows = [
        mode_qk56_quant(seed, L, D, D_FF),
        mode_qk28_quant(seed, L, D, D_FF),
        mode_calibrated(seed, L, D, D_FF, ACC_Q, ACC_K),
    ]
    for r in rows:
        print_row(r)
        print()

    c = rows[2]
    print("Calibrated scales (frozen from seed=1 L=32 empirical raw std):")
    print(f"  raw std Q={c['raw_std_Q']:.3f} K={c['raw_std_K']:.3f}")
    print(f"  ACC_SCALE_Q={c['ACC_SCALE_Q']:.8f} ACC_SCALE_K={c['ACC_SCALE_K']:.8f}")
    print(f"  QUANT_SCALE={QUANT:.8f}")


if __name__ == "__main__":
    main()
