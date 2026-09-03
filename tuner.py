"""Texel-style tuning: fit eval weights to self-play game outcomes.

Model: P(white wins | position) ~= sigmoid(K * eval_white_pov(position))
Since our `eval_cp` returns a side-to-move-relative score, everything below
converts to/from that consistently.

The eval is linear in the tunable weights once phase and all the frozen
terms are precomputed per position (see feature_extractor.py), so this is
literally a logistic regression with ~803 features, fit with minibatch Adam
and a touch of L2 shrinkage back towards the current shipped constants
(pure numpy, no torch/sklearn dependency).

Usage:
    python tuner.py --data data/*.jsonl --out tuned_params.npy \
        --epochs 40 --lr 0.02

Then:
    python apply_params.py tuned_params.npy > new_constants.py
and paste the relevant blocks into your engine file.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import chess

import feature_extractor as fe


def load_dataset(patterns: list[str]):
    fens, results = [], []
    for pattern in patterns:
        for path in sorted(glob.glob(pattern)):
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    fens.append(rec["fen"])
                    results.append(float(rec["result"]))
    if not fens:
        raise SystemExit(f"no data found for patterns {patterns}")
    return fens, results


def build_matrices(fens: list[str], results: list[float]):
    """Precompute the linear-regression design matrix.

    Returns:
        X       (N, 2*R_DIM + 1)  -- features for [mg_w, eg_w, tempo]
        offset  (N,)              -- frozen-term contribution, already
                                      phase-weighted and sign-flipped
        y       (N,)              -- label from the side-to-move's
                                      perspective (1 = side to move won)
    """
    n = len(fens)
    X = np.zeros((n, fe.PARAM_DIM), dtype=np.float64)
    offset = np.zeros(n, dtype=np.float64)
    y = np.zeros(n, dtype=np.float64)

    for i, (fen, result) in enumerate(zip(fens, results)):
        board = chess.Board(fen)
        R, mg_frozen, eg_frozen, phase, white_to_move = fe.extract(board)
        sign = 1.0 if white_to_move else -1.0
        w_mg = phase / 24.0
        w_eg = (24 - phase) / 24.0

        X[i, :fe.R_DIM] = sign * w_mg * R
        X[i, fe.R_DIM:2 * fe.R_DIM] = sign * w_eg * R
        X[i, 2 * fe.R_DIM] = 1.0  # tempo: always +1, added regardless of side

        offset[i] = sign * (mg_frozen * w_mg + eg_frozen * w_eg)
        y[i] = result if white_to_move else (1.0 - result)

        if (i + 1) % 50000 == 0:
            print(f"  extracted {i + 1}/{n}", flush=True)

    return X, offset, y


def fit_k(X: np.ndarray, offset: np.ndarray, y: np.ndarray, params0: np.ndarray) -> float:
    """1-D search for the sigmoid scale K that best fits the CURRENT params.

    Standard first step in Texel tuning: fix K once against the untouched
    eval, then hold it fixed while the weights are optimized. Re-running this
    after tuning and comparing is a good sanity check but not required.
    """
    raw = X @ params0 + offset
    best_k, best_loss = None, float("inf")
    for k in np.geomspace(1e-4, 1e-1, 60):
        p = 1.0 / (1.0 + np.exp(-k * raw))
        p = np.clip(p, 1e-9, 1 - 1e-9)
        loss = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
        if loss < best_loss:
            best_loss, best_k = loss, k
    print(f"fitted K = {best_k:.6g} (loss {best_loss:.4f})")
    return float(best_k)


def train(
    X: np.ndarray,
    offset: np.ndarray,
    y: np.ndarray,
    params0: np.ndarray,
    k: float,
    epochs: int,
    lr: float,
    batch_size: int,
    l2: float,
    seed: int = 0,
) -> np.ndarray:
    n = X.shape[0]
    params = params0.copy()

    # Adam state
    m = np.zeros_like(params)
    v = np.zeros_like(params)
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    t = 0

    rng = np.random.default_rng(seed)

    for epoch in range(epochs):
        order = rng.permutation(n)
        total_loss = 0.0
        for start in range(0, n, batch_size):
            idx = order[start:start + batch_size]
            xb, ob, yb = X[idx], offset[idx], y[idx]

            raw = xb @ params + ob
            pred = 1.0 / (1.0 + np.exp(-k * raw))
            pred_c = np.clip(pred, 1e-9, 1 - 1e-9)
            loss = -np.mean(yb * np.log(pred_c) + (1 - yb) * np.log(1 - pred_c))
            loss += l2 * np.sum((params - params0) ** 2) / len(params)
            total_loss += loss * len(idx)

            # d(cross-entropy)/d(raw) = k * (pred - y); chain rule through xb.
            grad_raw = k * (pred - yb) / len(idx)
            grad = xb.T @ grad_raw + 2 * l2 * (params - params0) / len(params)

            t += 1
            m = beta1 * m + (1 - beta1) * grad
            v = beta2 * v + (1 - beta2) * (grad ** 2)
            m_hat = m / (1 - beta1 ** t)
            v_hat = v / (1 - beta2 ** t)
            params -= lr * m_hat / (np.sqrt(v_hat) + eps)

        print(f"epoch {epoch + 1}/{epochs}  loss={total_loss / n:.5f}", flush=True)

    return params


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, nargs="+", required=True,
                     help="glob pattern(s) for .jsonl files from selfplay.py")
    ap.add_argument("--out", type=str, default="tuned_params.npy")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--l2", type=float, default=1e-6)
    ap.add_argument("--k", type=float, default=None,
                     help="fix the sigmoid scale instead of fitting it")
    args = ap.parse_args()

    fens, results = load_dataset(args.data)
    print(f"loaded {len(fens)} positions")

    print("extracting features...")
    X, offset, y = build_matrices(fens, results)

    params0 = fe.default_params()
    k = args.k if args.k is not None else fit_k(X, offset, y, params0)

    params = train(
        X, offset, y, params0, k,
        epochs=args.epochs, lr=args.lr, batch_size=args.batch_size, l2=args.l2,
    )

    np.save(args.out, params)
    print(f"saved tuned params -> {args.out}")

    # Quick before/after loss comparison on the training set.
    for name, p in (("before", params0), ("after", params)):
        raw = X @ p + offset
        pred = np.clip(1.0 / (1.0 + np.exp(-k * raw)), 1e-9, 1 - 1e-9)
        loss = -np.mean(y * np.log(pred) + (1 - y) * np.log(1 - pred))
        print(f"{name}: log-loss={loss:.5f}")


if __name__ == "__main__":
    main()
