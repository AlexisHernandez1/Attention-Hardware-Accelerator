#!/usr/bin/env python3
"""Full seed×L comparison: old default / QK=28 / robust calibrated+score dequant."""
from __future__ import annotations

import math
import importlib.util
from pathlib import Path

import numpy as np

SPEC = importlib.util.spec_from_file_location(
    "div", str(Path(__file__).resolve().parent / "diversity_check.py")
)
div = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(div)

QUANT = 1.0 / 128.0
# Robust max-|acc|→108 scales (seeds 1–5, L∈{16..256}, QK_W=56)
ACC_Q = 0.0059645441
ACC_K = 0.0052780764
SCORE_DEQUANT = 1.0 / 6.0
SEEDS = range(1, 6)
LS = [16, 32, 64, 128, 256]


def raw_mm(a, b, transpose_b=False):
    bb = b.T if transpose_b else b
    return a.astype(np.int32) @ bb.astype(np.int32)


def scale_clip(acc, scale):
    scaled = acc.astype(np.float64) * scale
    rounded = np.where(scaled >= 0, np.floor(scaled + 0.5), np.ceil(scaled - 0.5))
    return np.clip(rounded, -128, 127).astype(np.int16)


def pipeline(seed, L, D, D_FF, qk_w, acc_q, acc_k, score_dequant):
    t = div.tensors_mix(seed, L, D, D_FF, qk_w)
    Q = scale_clip(raw_mm(t["X"], t["W_q"]), acc_q)
    K = scale_clip(raw_mm(t["X"], t["W_k"]), acc_k)
    scores = scale_clip(raw_mm(Q, K, True), QUANT / math.sqrt(D))
    Sf = scores.astype(np.float64) * score_dequant
    sm = div.softmax_rows(Sf)
    H = div.row_entropy(sm)
    q_sat = bool(np.any(np.abs(Q) >= 127))  # detect rail; 127 or 128 both count
    # True saturation: value hit elem min/max after clip from larger pre-clip
    Qp = raw_mm(t["X"], t["W_q"]).astype(np.float64) * acc_q
    Kp = raw_mm(t["X"], t["W_k"]).astype(np.float64) * acc_k
    q_sat = bool(np.any(np.abs(Qp) >= 127.0 - 1e-9))
    k_sat = bool(np.any(np.abs(Kp) >= 127.0 - 1e-9))
    # Match C: SATURATION if min==elem_t_min or max==elem_t_max after store
    q_sat = bool(Q.min() == -128 or Q.max() == 127)
    k_sat = bool(K.min() == -128 or K.max() == 127)
    return {
        "Q_max_abs": int(np.max(np.abs(Q))),
        "K_max_abs": int(np.max(np.abs(K))),
        "sat": q_sat or k_sat,
        "score_std": float((scores.astype(np.float64) * QUANT).std()),
        "score_range": float((scores.max() - scores.min()) * QUANT),
        "score_raw": (int(scores.min()), int(scores.max())),
        "H": float(H.mean()),
        "dH": float(math.log(L) - H.mean()),
        "maxp": float(sm.max(axis=1).mean()),
        "uniform": 1.0 / L,
    }


def run_mode(name, qk_w, acc_q, acc_k, score_dequant):
    print(f"\n=== {name} ===")
    print(
        "L  seed  sat  Qabs  Kabs  score_std  score_rng  H      dH     maxp   1/L"
    )
    sat_n = 0
    n = 0
    for L in LS:
        for seed in SEEDS:
            r = pipeline(seed, L, 16, 64, qk_w, acc_q, acc_k, score_dequant)
            n += 1
            sat_n += int(r["sat"])
            print(
                f"{L:<3}{seed:<6}{'Y' if r['sat'] else 'N':<5}"
                f"{r['Q_max_abs']:<6}{r['K_max_abs']:<6}"
                f"{r['score_std']:<10.4f}{r['score_range']:<10.4f}"
                f"{r['H']:<7.4f}{r['dH']:<7.4f}{r['maxp']:<7.4f}{r['uniform']:.4f}"
            )
    print(f"saturation: {sat_n}/{n}")


def main():
    print("Calibration constants:")
    print(f"  ACC_SCALE_Q={ACC_Q:.10f}  ACC_SCALE_K={ACC_K:.10f}")
    print(f"  SCORE_DEQUANT_SCALE={SCORE_DEQUANT:.10f}  SCORE_SCALE=QUANT/sqrt(D)")
    run_mode("A: QK=56 + ACC=QUANT + score_dequant=QUANT (old default)", 56, QUANT, QUANT, QUANT)
    run_mode("B: QK=28 + ACC=QUANT + score_dequant=QUANT", 28, QUANT, QUANT, QUANT)
    run_mode(
        "C: QK=56 + robust ACC_Q/K (max→108) + SCORE_DEQUANT=1/6",
        56, ACC_Q, ACC_K, SCORE_DEQUANT,
    )


if __name__ == "__main__":
    main()
