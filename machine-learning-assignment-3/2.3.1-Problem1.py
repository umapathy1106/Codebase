"""
Problem 1 — California Housing: splitting, standardization, and Fig. 0.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")          # non-interactive backend (saves to file)
from sklearn.datasets import fetch_california_housing

# ── Reproducibility ──────────────────────────────────────────────────────────
np.random.seed(42)

# ── Load full dataset ─────────────────────────────────────────────────────────
_housing  = fetch_california_housing()
_X_all    = _housing.data[:, 0]   # MedInc only, shape (20640,)
_y_all    = _housing.target        # median house value in $100k, shape (20640,)
_N        = _X_all.shape[0]        # 20640

# Single global permutation (seed already set above)
_GLOBAL_IDX = np.random.permutation(_N)
_TEST_IDX   = _GLOBAL_IDX[10000:15000]   # fixed test set, always the same

# ── i. split_data ─────────────────────────────────────────────────────────────

def split_data(x, y, train_size):
    """
    Return (x_train, x_test, y_train, y_test) using the single global
    permutation of indices computed with numpy.random.seed(42).

    Training : indices [0 : train_size]
    Test      : indices [10000 : 15000]  (fixed, size 5000)

    Parameters
    ----------
    x          : 1-D array of features (length N = 20640)
    y          : 1-D array of targets  (length N = 20640)
    train_size : int, must be <= 10000

    Raises
    ------
    ValueError if train_size > 10000
    """
    if train_size > 10000:
        raise ValueError(
            f"train_size={train_size} exceeds the maximum allowed value of 10000."
        )

    train_idx = _GLOBAL_IDX[0:train_size]

    x_train = x[train_idx]
    x_test  = x[_TEST_IDX]
    y_train = y[train_idx]
    y_test  = y[_TEST_IDX]

    return x_train, x_test, y_train, y_test


# ── ii. Verify sizes ──────────────────────────────────────────────────────────

print("=== Size Verification ===")
for ts in [1000, 5000, 10000]:
    xt, xe, yt, ye = split_data(_X_all, _y_all, ts)
    print(f"  train_size={ts:>5d}:  x_train={xt.shape}  x_test={xe.shape}  "
          f"y_train={yt.shape}  y_test={ye.shape}")
    assert xt.shape == (ts,),    f"x_train shape mismatch for train_size={ts}"
    assert xe.shape == (5000,),  f"x_test shape mismatch for train_size={ts}"
    assert yt.shape == (ts,),    f"y_train shape mismatch for train_size={ts}"
    assert ye.shape == (5000,),  f"y_test shape mismatch for train_size={ts}"
print("  All size checks passed.\n")


# ── iii. standardize(v) ───────────────────────────────────────────────────────

def standardize(v):
    """
    Z-score normalization: z = (v - mu) / sigma
    where mu = mean(v), sigma = std(v) [population, ddof=0].

    Parameters
    ----------
    v : 1-D or 2-D ndarray (training data)

    Returns
    -------
    v_z   : standardized array (same shape)
    mu    : mean  (scalar or 1-D array)
    sigma : std   (scalar or 1-D array)
    """
    mu    = v.mean(axis=0)
    sigma = v.std(axis=0)
    v_z   = (v - mu) / sigma
    return v_z, mu, sigma


# Apply standardization to features only (train statistics → both sets)
x_train_raw, x_test_raw, y_train, y_test = split_data(_X_all, _y_all, 5000)

x_train_z, mu_x, sigma_x = standardize(x_train_raw)
x_test_z = (x_test_raw - mu_x) / sigma_x    # same transform, no re-fit

print("=== Standardization (train_size=5000) ===")
print(f"  MedInc training mean  (µ)  : {mu_x:.4f}  (raw $10k units)")
print(f"  MedInc training std   (σ)  : {sigma_x:.4f}")
print(f"  x_train_z  — mean: {x_train_z.mean():.2e}  std: {x_train_z.std():.6f}")
print(f"  x_test_z   — mean: {x_test_z.mean():.4f}  std: {x_test_z.std():.4f}")
print(f"  y unchanged — mean: {y_train.mean():.4f}  std: {y_train.std():.4f}\n")


# ── iv. Fig. 0 ────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Fig. 0 — California Housing (train_size = 5 000)", fontsize=13)

# Left: unstandardized
ax = axes[0]
ax.scatter(x_train_raw, y_train, s=6, alpha=0.4, color="steelblue", rasterized=True)
ax.set_xlabel("MedInc — median income ($10k)", fontsize=11)
ax.set_ylabel("Median house value ($100k)", fontsize=11)
ax.set_title("Unstandardized features")
ax.grid(True, linewidth=0.4)

# Right: standardized
ax = axes[1]
ax.scatter(x_train_z, y_train, s=6, alpha=0.4, color="darkorange", rasterized=True)
ax.set_xlabel("MedInc — standardized (z-score)", fontsize=11)
ax.set_ylabel("Median house value ($100k)", fontsize=11)
ax.set_title("Standardized features (y unchanged)")
ax.grid(True, linewidth=0.4)

plt.tight_layout()

import os
out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fig0_scatter.png")
plt.savefig(out_path, dpi=150)
print(f"Fig. 0 saved to: {out_path}")
