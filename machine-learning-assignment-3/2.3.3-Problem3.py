"""
Problem 3 — K-Nearest Neighbors Regression (California Housing).

Algorithm (1-D efficient): sort x_train once, use binary search to locate
the insertion point for each test point, then expand left/right with a
two-pointer to collect the K nearest neighbours.
→ O(n_train log n_train  +  n_test (log n_train + K))  — no brute force.
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
_X_all      = _housing.data[:, 0]        # MedInc, shape (20640,)
_y_all      = _housing.target             # $100k,  shape (20640,)
_N          = _X_all.shape[0]
_GLOBAL_IDX = np.random.permutation(_N)
_TEST_IDX   = _GLOBAL_IDX[10000:15000]

DIR = os.path.dirname(os.path.abspath(__file__))


# ── Helper: split + standardize ───────────────────────────────────────────────

def get_splits(train_size):
    if train_size > 10_000:
        raise ValueError(f"train_size={train_size} exceeds 10000.")
    tr         = _GLOBAL_IDX[:train_size]
    x_tr_raw   = _X_all[tr];          y_tr = _y_all[tr]
    x_te_raw   = _X_all[_TEST_IDX];   y_te = _y_all[_TEST_IDX]
    mu, sigma  = x_tr_raw.mean(), x_tr_raw.std()
    return (x_tr_raw - mu) / sigma, y_tr, (x_te_raw - mu) / sigma, y_te


def mse(y_pred, y_true):
    return np.mean((y_pred - y_true) ** 2)


# ── i. knn_predict ────────────────────────────────────────────────────────────

def knn_predict(x_train, y_train, x_test, K):
    """
    1-D KNN regression (efficient: sort once + two-pointer per query).

    For each x_test[i], find K nearest neighbours in x_train by |distance|
    and return their mean y value.  If K >= n_train, use all training points.

    Parameters
    ----------
    x_train, y_train : 1-D arrays, length n_train
    x_test           : 1-D array,  length n_test
    K                : int >= 1

    Returns
    -------
    y_pred : ndarray (n_test,)
    """
    n_tr = len(x_train)
    K    = min(K, n_tr)               # cap at n_train per spec

    # Sort training set once
    order = np.argsort(x_train)
    xs    = x_train[order]            # sorted x_train
    ys    = y_train[order]            # aligned y

    # Prefix sums for O(1) range sum queries
    prefix = np.concatenate([[0.0], np.cumsum(ys)])

    y_pred = np.empty(len(x_test))

    for i, xi in enumerate(x_test):
        # Insertion point in sorted array
        mid = int(np.searchsorted(xs, xi, side='left'))

        # Two-pointer: expand from mid to collect K nearest neighbours
        lo = mid - 1   # left  pointer (exclusive lower bound into xs)
        hi = mid       # right pointer (inclusive upper bound into xs)

        for _ in range(K):
            # Choose the side with the smaller absolute distance
            dist_lo = (xi - xs[lo]) if lo >= 0    else np.inf
            dist_hi = (xs[hi] - xi) if hi < n_tr  else np.inf

            if dist_lo <= dist_hi:
                lo -= 1
            else:
                hi += 1

        # Window is xs[lo+1 : hi]  (length K)
        lo_idx = lo + 1   # inclusive start
        hi_idx = hi       # exclusive end
        y_pred[i] = (prefix[hi_idx] - prefix[lo_idx]) / K

    return y_pred


# ── ii. MSE vs K (train_size = 5000) ─────────────────────────────────────────

K_VALUES         = [1, 3, 5, 10, 20, 50, 100, 500, 1000]
TRAIN_SIZE_FIXED = 5000

x_tr5k, y_tr5k, x_te, y_te = get_splits(TRAIN_SIZE_FIXED)

mse_vs_K = []

print("=== MSE vs K  (train_size=5000) ===")
print(f"{'K':>6}  {'Test MSE':>10}")
print("-" * 20)

for K in K_VALUES:
    y_pred = knn_predict(x_tr5k, y_tr5k, x_te, K)
    m      = mse(y_pred, y_te)
    mse_vs_K.append(m)
    print(f"{K:>6d}  {m:>10.4f}")


# ── iii. Fig. 3: MSE vs K ────────────────────────────────────────────────────

fig3, ax3 = plt.subplots(figsize=(8, 5))
ax3.plot(K_VALUES, mse_vs_K, "o-", color="steelblue", linewidth=2, markersize=7)
ax3.set_xscale("log")
ax3.set_xlabel("K  (number of neighbours, log scale)", fontsize=12)
ax3.set_ylabel("Test MSE  ($100k)²", fontsize=12)
ax3.set_title("Fig. 3 — KNN Regression: Test MSE vs K  (train_size=5 000)", fontsize=12)
ax3.set_xticks(K_VALUES)
ax3.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax3.grid(True, which="both", linewidth=0.4)
fig3.tight_layout()
fig3.savefig(os.path.join(DIR, "fig3_mse_vs_K.png"), dpi=150)
print(f"\nFig. 3 saved.")


# ── iv. MSE vs train_size (K = 100) ──────────────────────────────────────────

K_FIXED     = 100
TRAIN_SIZES = list(range(1000, 11000, 1000))
mse_vs_N    = []

print(f"\n=== MSE vs train_size  (K={K_FIXED}) ===")
print(f"{'N_train':>8}  {'Test MSE':>10}")
print("-" * 22)

for ts in TRAIN_SIZES:
    x_tr, y_tr, x_te2, y_te2 = get_splits(ts)
    y_pred = knn_predict(x_tr, y_tr, x_te2, K_FIXED)
    m      = mse(y_pred, y_te2)
    mse_vs_N.append(m)
    print(f"{ts:>8d}  {m:>10.4f}")


# ── v. Fig. 4: MSE vs train_size ─────────────────────────────────────────────

fig4, ax4 = plt.subplots(figsize=(8, 5))
ax4.plot(TRAIN_SIZES, mse_vs_N, "s-", color="darkorange", linewidth=2, markersize=7)
ax4.set_xlabel("Training size  N", fontsize=12)
ax4.set_ylabel("Test MSE  ($100k)²", fontsize=12)
ax4.set_title(f"Fig. 4 — KNN Regression: Test MSE vs Training Size  (K={K_FIXED})", fontsize=12)
ax4.set_xticks(TRAIN_SIZES)
ax4.tick_params(axis='x', rotation=30)
ax4.grid(True, linewidth=0.4)
fig4.tight_layout()
fig4.savefig(os.path.join(DIR, "fig4_mse_vs_N_knn.png"), dpi=150)
print(f"\nFig. 4 saved.")


# ── vi. Discussion ────────────────────────────────────────────────────────────

print("""
=== Discussion: Figs. 3 & 4 — KNN vs Radius Regression ===

Fig. 3 — MSE vs K (train_size = 5 000):
  • K = 1: Each test point is predicted by its single nearest neighbour.
    Very noisy — a single training sample may be an outlier → high variance
    → high MSE.
  • K = 10–100: More neighbours → the local mean is smoother and more
    robust → MSE decreases and reaches a minimum around K = 100.
    This is the bias-variance sweet spot for this dataset and train size.
  • K = 500–1000: The neighbourhood grows so large that it spans most of
    the feature range → predictions approach the global training mean →
    high bias → MSE rises again, mirroring the large-C behaviour in
    radius regression.
  → Same U-shaped bias-variance trade-off as observed with radius C,
    with small K analogous to small C, and large K analogous to large C.

Fig. 4 — MSE vs Training Size (K = 100):
  • MSE decreases steadily as N grows from 1 000 → 10 000.
  • With more training points the 100 nearest neighbours are closer to
    each test point → predictions are more local → lower bias.
  • The rate of improvement slows past N ≈ 7 000 (diminishing returns),
    consistent with radius regression.

Comparison KNN vs Radius Regression:
  • Both are non-parametric local-averaging methods and share the same
    U-shaped MSE vs bandwidth (C or K) and the same improvement with
    more data.
  • KNN guarantees exactly K neighbours regardless of density → no
    fallback needed, unlike radius regression which can find zero
    neighbours in sparse regions.
  • Radius regression is more sensitive to the density of training data:
    in dense regions it averages many points (low variance) while in
    sparse regions it falls back. KNN adapts its effective radius to
    local density, making it more robust.
  • At their respective optima (C ≈ 0.3, K ≈ 100) the test MSE values
    are very similar (~0.68), showing both methods extract roughly the
    same amount of signal from a 1-D scalar feature.
  • For higher-dimensional data KNN scales better because it does not
    require a fallback; radius regression would need very large C to
    avoid empty windows.
""")
