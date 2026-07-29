#!/usr/bin/env python3
"""Weight / QK / softmax diversity check: mix_seed vs legacy shared PRNG.

Mirrors transformer_block_test.c PRNG + int8 matmul ACC_SCALE contract so we
can quantify non-degeneracy without dumping full tensors from Spike.
"""
from __future__ import annotations

import argparse
import math
from itertools import combinations
from typing import Dict, Tuple

import numpy as np

QUANT_SCALE = 1.0 / 128.0
INPUT_RAW_MAGNITUDE = 64
QK_WEIGHT_RAW_MAGNITUDE = 28
OTHER_WEIGHT_RAW_MAGNITUDE = 8
TAGS = {
    "X": 0x58,
    "W_q": 0x51,
    "W_k": 0x4B,
    "W_v": 0x56,
    "W_o": 0x4F,
    "W_up": 0x55,
    "W_down": 0x44,
}


def mix_seed(seed: int, tag: int) -> int:
    x = (seed ^ tag) & 0xFFFFFFFF
    x = (x * 0x9E3779B9) & 0xFFFFFFFF
    x ^= x >> 16
    x = (x * 0x85EBCA6B) & 0xFFFFFFFF
    x ^= x >> 13
    return x if x else 1


def fill_random(n: int, magnitude: int, seed: int) -> np.ndarray:
    state = seed & 0xFFFFFFFF
    out = np.empty(n, dtype=np.int16)
    span = 2 * magnitude + 1
    for i in range(n):
        state = (state * 1664525 + 1013904223) & 0xFFFFFFFF
        out[i] = int(state % span) - magnitude
    return out


def fill_random_continue(n: int, magnitude: int, state: int) -> Tuple[np.ndarray, int]:
    out = np.empty(n, dtype=np.int16)
    span = 2 * magnitude + 1
    for i in range(n):
        state = (state * 1664525 + 1013904223) & 0xFFFFFFFF
        out[i] = int(state % span) - magnitude
    return out, state


def tensors_mix(seed: int, L: int, D: int, D_FF: int, qk_w: int) -> Dict[str, np.ndarray]:
    return {
        "X": fill_random(L * D, INPUT_RAW_MAGNITUDE, mix_seed(seed, TAGS["X"])).reshape(L, D),
        "W_q": fill_random(D * D, qk_w, mix_seed(seed, TAGS["W_q"])).reshape(D, D),
        "W_k": fill_random(D * D, qk_w, mix_seed(seed, TAGS["W_k"])).reshape(D, D),
        "W_v": fill_random(D * D, OTHER_WEIGHT_RAW_MAGNITUDE, mix_seed(seed, TAGS["W_v"])).reshape(D, D),
        "W_o": fill_random(D * D, 2 * OTHER_WEIGHT_RAW_MAGNITUDE, mix_seed(seed, TAGS["W_o"])).reshape(D, D),
        "W_up": fill_random(D * D_FF, OTHER_WEIGHT_RAW_MAGNITUDE, mix_seed(seed, TAGS["W_up"])).reshape(D, D_FF),
        "W_down": fill_random(D_FF * D, OTHER_WEIGHT_RAW_MAGNITUDE, mix_seed(seed, TAGS["W_down"])).reshape(D_FF, D),
    }


def tensors_shared(seed: int, L: int, D: int, D_FF: int, qk_w: int) -> Dict[str, np.ndarray]:
    """Legacy: one continuous LCG stream in initialize order (X then weights)."""
    state = seed & 0xFFFFFFFF
    X, state = fill_random_continue(L * D, INPUT_RAW_MAGNITUDE, state)
    W_q, state = fill_random_continue(D * D, qk_w, state)
    W_k, state = fill_random_continue(D * D, qk_w, state)
    W_v, state = fill_random_continue(D * D, OTHER_WEIGHT_RAW_MAGNITUDE, state)
    W_o, state = fill_random_continue(D * D, 2 * OTHER_WEIGHT_RAW_MAGNITUDE, state)
    W_up, state = fill_random_continue(D * D_FF, OTHER_WEIGHT_RAW_MAGNITUDE, state)
    W_down, state = fill_random_continue(D_FF * D, OTHER_WEIGHT_RAW_MAGNITUDE, state)
    return {
        "X": X.reshape(L, D),
        "W_q": W_q.reshape(D, D),
        "W_k": W_k.reshape(D, D),
        "W_v": W_v.reshape(D, D),
        "W_o": W_o.reshape(D, D),
        "W_up": W_up.reshape(D, D_FF),
        "W_down": W_down.reshape(D_FF, D),
    }


def int8_matmul(a: np.ndarray, b: np.ndarray, acc_scale: float, transpose_b: bool = False) -> np.ndarray:
    bb = b.T if transpose_b else b
    acc = a.astype(np.int32) @ bb.astype(np.int32)
    scaled = acc.astype(np.float64) * acc_scale
    rounded = np.where(scaled >= 0, np.floor(scaled + 0.5), np.ceil(scaled - 0.5))
    return np.clip(rounded, -128, 127).astype(np.int16)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    av = a.astype(np.float64).ravel()
    bv = b.astype(np.float64).ravel()
    na = np.linalg.norm(av)
    nb = np.linalg.norm(bv)
    if na == 0 or nb == 0:
        return float("nan")
    return float(np.dot(av, bv) / (na * nb))


def max_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))))


def softmax_rows(scores_dequant: np.ndarray) -> np.ndarray:
    x = scores_dequant - scores_dequant.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)


def row_entropy(p: np.ndarray) -> np.ndarray:
    # natural log; ignore zeros
    out = np.zeros(p.shape[0], dtype=np.float64)
    for i in range(p.shape[0]):
        row = p[i]
        row = row[row > 0]
        out[i] = float(-(row * np.log(row)).sum())
    return out


def analyze(name: str, tensors: Dict[str, np.ndarray], D: int) -> None:
    weights = ["W_q", "W_k", "W_v", "W_o", "W_up", "W_down"]
    print(f"\n=== {name}: weight pairwise similarity ===")
    print(f"{'pair':<16} {'cosine':>10} {'max|diff|':>12}")
    for a, b in combinations(weights, 2):
        wa, wb = tensors[a], tensors[b]
        if wa.shape != wb.shape:
            # compare flattened overlap of equal-length only for square pairs already filtered
            # skip unequal shapes (W_up / W_down vs square)
            continue
        print(f"{a+' vs '+b:<16} {cosine(wa, wb):10.4f} {max_abs_diff(wa, wb):12.1f}")

    # Also report unequal-shape pairs via flattened cosine after zero-pad to common length? Skip.
    # Explicit W_up vs W_down transpose shapes differ — flatten and truncate to min length for a coarse check.
    for a, b in [("W_up", "W_down"), ("W_q", "W_up"), ("W_k", "W_up")]:
        av = tensors[a].astype(np.float64).ravel()
        bv = tensors[b].astype(np.float64).ravel()
        n = min(av.size, bv.size)
        c = float(np.dot(av[:n], bv[:n]) / (np.linalg.norm(av[:n]) * np.linalg.norm(bv[:n])))
        mad = float(np.max(np.abs(av[:n] - bv[:n])))
        print(f"{a+' vs '+b+' (trunc)':<16} {c:10.4f} {mad:12.1f}")

    Q = int8_matmul(tensors["X"], tensors["W_q"], QUANT_SCALE)
    K = int8_matmul(tensors["X"], tensors["W_k"], QUANT_SCALE)
    score_scale = QUANT_SCALE / math.sqrt(D)
    scores = int8_matmul(Q, K, score_scale, transpose_b=True)
    scores_f = scores.astype(np.float64) * QUANT_SCALE  # stored at QUANT_SCALE like other tensors
    # Attention scores in the C test use SCORE_SCALE as ACC_SCALE, then stored as int8;
    # dequantize for softmax uses QUANT_SCALE (dequantize()).
    print(f"\n=== {name}: QK scores (post int8, dequantized with QUANT_SCALE) ===")
    print(f"Q max_abs={int(np.max(np.abs(Q)))}  K max_abs={int(np.max(np.abs(K)))}  "
          f"sat_Q={bool(np.any((Q <= -128) | (Q >= 127)))}  sat_K={bool(np.any((K <= -128) | (K >= 127)))}")
    print(f"scores raw range=[{int(scores.min())}, {int(scores.max())}]  "
          f"std={scores_f.std():.6f}  range(max-min)={float(scores_f.max() - scores_f.min()):.6f}")

    weights_sm = softmax_rows(scores_f)
    H = row_entropy(weights_sm)
    H_uniform = math.log(scores.shape[1])
    print(f"\n=== {name}: softmax rows ===")
    print(f"uniform entropy ln(L)={H_uniform:.4f}  mean row entropy={H.mean():.4f}  "
          f"min={H.min():.4f}  max={H.max():.4f}")
    print(f"mean row max_prob={weights_sm.max(axis=1).mean():.6f}  "
          f"(uniform 1/L={1.0 / scores.shape[1]:.6f})")
    for row in (0, scores.shape[0] // 2, scores.shape[0] - 1):
        p = weights_sm[row]
        print(f"  row {row}: max={p.max():.6f}  entropy={H[row]:.4f}  "
              f"top5={np.sort(p)[-5:][::-1]}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--L", type=int, default=32)
    ap.add_argument("--D", type=int, default=16)
    ap.add_argument("--D_FF", type=int, default=64)
    ap.add_argument("--qk-w", type=int, default=QK_WEIGHT_RAW_MAGNITUDE)
    args = ap.parse_args()

    print(f"Diversity check: seed={args.seed} L={args.L} D={args.D} D_FF={args.D_FF} QK_W={args.qk_w}")
    mix = tensors_mix(args.seed, args.L, args.D, args.D_FF, args.qk_w)
    shared = tensors_shared(args.seed, args.L, args.D, args.D_FF, args.qk_w)

    # Sanity: mix_seed W_k identical at two L values
    mix16 = tensors_mix(args.seed, 16, args.D, args.D_FF, args.qk_w)
    mix32 = tensors_mix(args.seed, 32, args.D, args.D_FF, args.qk_w)
    print(f"mix_seed W_k identical L=16 vs L=32: {np.array_equal(mix16['W_k'], mix32['W_k'])}")
    shared16 = tensors_shared(args.seed, 16, args.D, args.D_FF, args.qk_w)
    shared32 = tensors_shared(args.seed, 32, args.D, args.D_FF, args.qk_w)
    print(f"shared-PRNG W_k identical L=16 vs L=32: {np.array_equal(shared16['W_k'], shared32['W_k'])}")

    analyze("mix_seed (current)", mix, args.D)
    analyze("shared-PRNG (legacy reconstruction)", shared, args.D)


if __name__ == "__main__":
    main()
