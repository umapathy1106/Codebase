"""
Problem 2 — Radius-Based Regression (California Housing).

Algorithm: sort x_train once, use binary search + prefix sums for O(n_test log n_train)
predictions — avoids O(n_train * n_test) brute force per the Conventions.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

from sklearn.datasets import fetch_california_housing

# ── Reproducibility & data ────────────────────────────────────────────────────
np.random.seed(42)

_housing    = fetch_california_housing()
_X_all      = _housing.data[:, 0]   # MedInc, shape (20640,)
_y_all      = _housing.target        # $100k,  shape (20640,)
_N          = _X_all.shape[0]
_GLOBAL_IDX = np.random.permutation(_N)
_TEST_IDX   = _GLOBAL_IDX[10000:15000]

DIR = os.path.dirname(os.path.abspath(__file__))


# ── Helper: split + standardize ───────────────────────────────────────────────

def get_splits(train_size):
    """Return (x_train_z, y_train, x_test_z, y_test) for given train_size."""
    if train_size > 10_000:
        raise ValueError(f"train_size={train_size} exceeds 10000.")
    tr = _GLOBAL_IDX[:train_size]
    x_tr_raw = _X_all[tr];   y_tr = _y_all[tr]
    x_te_raw = _X_all[_TEST_IDX]; y_te = _y_all[_TEST_IDX]
    mu    = x_tr_raw.mean()
    sigma = x_tr_raw.std()
    return (x_tr_raw - mu) / sigma, y_tr, (x_te_raw - mu) / sigma, y_te


# ── i. radius_predict ─────────────────────────────────────────────────────────

def radius_predict(x_train, y_train, x_test, C):
    """
    Radius-based regression (1-D, efficient).

    For each x_test[i], predict mean(y_train[j] for |x_train[j]-x_test[i]| <= C).
    Fallback: mean(y_train) if no neighbour found.

    Uses sorting + prefix sums → O(n_train log n_train + n_test log n_train).

    Returns
    -------
    y_pred     : ndarray (n_test,)
    fallback   : bool ndarray (n_test,) — True where fallback was used
    """
    # Sort training set by x value once
    order   = np.argsort(x_train)
    xs      = x_train[order]         # sorted x_train
    ys      = y_train[order]         # y aligned to sorted x

    # Prefix sums for O(1) range mean queries
    prefix  = np.concatenate([[0.0], np.cumsum(ys)])   # length n_train+1
    n_tr    = len(xs)
    fallback_mean = y_train.mean()

    y_pred   = np.empty(len(x_test))
    fallback = np.zeros(len(x_test), dtype=bool)

    for i, xi in enumerate(x_test):
        lo = int(np.searchsorted(xs, xi - C, side='left'))
        hi = int(np.searchsorted(xs, xi + C, side='right'))  # exclusive
        count = hi - lo
        if count == 0:
            y_pred[i]   = fallback_mean
            fallback[i] = True
        else:
            y_pred[i] = (prefix[hi] - prefix[lo]) / count

    return y_pred, fallback


# ── ii. MSE vs C (train_size = 5000) ─────────────────────────────────────────

def mse(y_pred, y_true):
    return np.mean((y_pred - y_true) ** 2)

C_VALUES = [0.001, 0.005, 0.01, 0.05, 0.1, 0.3, 0.5, 1, 2, 3, 4, 5]
TRAIN_SIZE_FIXED = 5000

x_tr5k, y_tr5k, x_te, y_te = get_splits(TRAIN_SIZE_FIXED)

mse_vs_C       = []
fallback_frac  = []

print("=== MSE vs C  (train_size=5000) ===")
print(f"{'C':>8}  {'Test MSE':>10}  {'Fallback %':>10}")
print("-" * 33)

for C in C_VALUES:
    y_pred, fb = radius_predict(x_tr5k, y_tr5k, x_te, C)
    m  = mse(y_pred, y_te)
    ff = fb.mean() * 100
    mse_vs_C.append(m)
    fallback_frac.append(ff)
    print(f"{C:>8.3f}  {m:>10.4f}  {ff:>9.2f}%")

# ── iii. Fig. 1: MSE vs C ────────────────────────────────────────────────────

fig1, ax1 = plt.subplots(figsize=(8, 5))
ax1.plot(C_VALUES, mse_vs_C, "o-", color="steelblue", linewidth=2, markersize=7)
ax1.set_xscale("log")
ax1.set_xlabel("Radius C  (standardized x-scale, log)", fontsize=12)
ax1.set_ylabel("Test MSE  ($100k)²", fontsize=12)
ax1.set_title("Fig. 1 — Radius Regression: Test MSE vs C  (train_size=5 000)", fontsize=12)
ax1.grid(True, which="both", linewidth=0.4)
# annotate fallback fractions
for C, m, ff in zip(C_VALUES, mse_vs_C, fallback_frac):
    if ff > 0.5:
        ax1.annotate(f"{ff:.0f}% fb", (C, m), textcoords="offset points",
                     xytext=(6, 4), fontsize=8, color="crimson")
fig1.tight_layout()
fig1.savefig(os.path.join(DIR, "fig1_mse_vs_C.png"), dpi=150)
print(f"\nFig. 1 saved.")


# ── iv. MSE vs train_size (C = 0.1) ──────────────────────────────────────────

C_FIXED      = 0.1
TRAIN_SIZES  = list(range(1000, 11000, 1000))
mse_vs_N     = []

print(f"\n=== MSE vs train_size  (C={C_FIXED}) ===")
print(f"{'N_train':>8}  {'Test MSE':>10}")
print("-" * 22)

for ts in TRAIN_SIZES:
    x_tr, y_tr, x_te2, y_te2 = get_splits(ts)
    y_pred, _ = radius_predict(x_tr, y_tr, x_te2, C_FIXED)
    m = mse(y_pred, y_te2)
    mse_vs_N.append(m)
    print(f"{ts:>8d}  {m:>10.4f}")


# ── v. Fig. 2: MSE vs train_size ─────────────────────────────────────────────

fig2, ax2 = plt.subplots(figsize=(8, 5))
ax2.plot(TRAIN_SIZES, mse_vs_N, "s-", color="darkorange", linewidth=2, markersize=7)
ax2.set_xlabel("Training size  N", fontsize=12)
ax2.set_ylabel("Test MSE  ($100k)²", fontsize=12)
ax2.set_title(f"Fig. 2 — Radius Regression: Test MSE vs Training Size  (C={C_FIXED})", fontsize=12)
ax2.set_xticks(TRAIN_SIZES)
ax2.tick_params(axis='x', rotation=30)
ax2.grid(True, linewidth=0.4)
fig2.tight_layout()
fig2.savefig(os.path.join(DIR, "fig2_mse_vs_N.png"), dpi=150)
print(f"\nFig. 2 saved.")


# ── vi. Discussion ────────────────────────────────────────────────────────────

print("""
=== Discussion: Figs. 1 & 2 ===

Fig. 1 — MSE vs Radius C (train_size = 5 000):
  • Very small C (0.001–0.01): Many test points find no neighbour in the
    tiny window → high fallback rate → the predictor collapses to the
    global training mean, producing high MSE (high bias from fallback).
  • Moderate C (≈ 0.1–0.5): The window captures enough local neighbours to
    average out noise while still being local enough to follow the
    income–price signal → MSE reaches its minimum here.
  • Large C (≥ 1): The window widens to cover most of the training set,
    so the prediction approaches the global training mean for every test
    point → the model loses locality → MSE rises again (high bias).
  The U-shaped MSE curve is the classic bias–variance trade-off:
    - Small C → high variance + fallback bias.
    - Large C → high smoothing bias.
    - Optimal C balances the two.

Fallback impact:
  • At C = 0.001 and 0.005, a noticeable fraction of test points fall back
    to the global mean (annotated on Fig. 1). When the fallback fraction is
    large it dominates MSE, because the global mean is a poor local
    estimator for most test points.

Fig. 2 — MSE vs Training Size (C = 0.1):
  • As training size grows from 1 000 → 10 000, test MSE decreases
    monotonically (or near-monotonically).
  • More training points → the radius window C = 0.1 contains more
    neighbours → the local mean is a better estimate (lower variance).
  • Also fewer test points need the fallback as the training set becomes
    denser in input space.
  • The rate of improvement slows at larger N, consistent with the
    expected O(1/N) variance reduction of sample means.
""")
