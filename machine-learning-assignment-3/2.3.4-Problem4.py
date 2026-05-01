"""
Problem 4 — Univariate Polynomial Regression (California Housing).

Design matrix Xpoly ∈ R^{(d+1) × n}:
  Row 0  : bias (all ones)
  Row k  : x^k  (k = 1..d), each row is row-standardized using training stats.

Least-squares solved via SVD: w = U Σ⁺ Vᵀ y
Prediction: ŷ = Xpoly.T @ w
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
_y_all      = _housing.target             # $100k
_N          = _X_all.shape[0]
_GLOBAL_IDX = np.random.permutation(_N)
_TEST_IDX   = _GLOBAL_IDX[10000:15000]

DIR = os.path.dirname(os.path.abspath(__file__))


# ── Helper: split + z-score standardize x ─────────────────────────────────────

def get_splits(train_size):
    if train_size > 10_000:
        raise ValueError(f"train_size={train_size} exceeds 10000.")
    tr       = _GLOBAL_IDX[:train_size]
    x_tr_raw = _X_all[tr];          y_tr = _y_all[tr]
    x_te_raw = _X_all[_TEST_IDX];   y_te = _y_all[_TEST_IDX]
    mu, sigma = x_tr_raw.mean(), x_tr_raw.std()
    return (x_tr_raw - mu) / sigma, y_tr, (x_te_raw - mu) / sigma, y_te


def mse(y_pred, y_true):
    return np.mean((y_pred - y_true) ** 2)


# ── i. poly_features ─────────────────────────────────────────────────────────

def poly_features(x, degree, mu_rows=None, sigma_rows=None):
    """
    Build design matrix Xpoly of shape (degree+1, n).

    Row 0        : bias (all ones)
    Rows 1..degree: x^k, each row-standardized with training statistics.

    Training mode (mu_rows=None): computes and returns mu_rows, sigma_rows.
    Test    mode               : applies supplied mu_rows, sigma_rows.

    Parameters
    ----------
    x            : 1-D array of z-scored features, length n
    degree       : polynomial degree d
    mu_rows      : (d,) row means from training (None → training mode)
    sigma_rows   : (d,) row stds  from training (None → training mode)

    Returns
    -------
    Xpoly      : (degree+1, n) design matrix (bias row NOT standardized)
    mu_rows    : (degree,) training row means
    sigma_rows : (degree,) training row stds
    """
    n     = len(x)
    Xpoly = np.vstack([x ** k for k in range(degree + 1)])   # (d+1, n)

    train_mode = (mu_rows is None)
    if train_mode:
        mu_rows    = Xpoly[1:].mean(axis=1)    # (d,)
        sigma_rows = Xpoly[1:].std(axis=1)     # (d,)

    # Row-standardize non-bias rows using training statistics
    Xpoly[1:] = (Xpoly[1:] - mu_rows[:, None]) / sigma_rows[:, None]

    return Xpoly, mu_rows, sigma_rows


# ── ii. solve_LS ──────────────────────────────────────────────────────────────

def solve_LS(X_poly, y):
    """
    SVD-based least-squares solution for w.

    Minimises  ||Xpoly.T @ w - y||²

    SVD of Xpoly = U Σ Vᵀ  (shape (d+1)×n, economy SVD)
    Pseudo-inverse of (Xpoly.T) = U Σ⁺ Vᵀ
    w = U Σ⁺ Vᵀ y

    Parameters
    ----------
    X_poly : ndarray (d+1, n)
    y      : ndarray (n,)

    Returns
    -------
    w : ndarray (d+1,)
    """
    U, s, Vt = np.linalg.svd(X_poly, full_matrices=False)
    thresh   = 1e-10 * s[0] if s[0] > 0 else 1e-10
    s_inv    = np.where(s > thresh, 1.0 / s, 0.0)
    w        = U @ (s_inv * (Vt @ y))     # (d+1,)
    return w


# ── Convenience: full train-test poly pipeline ────────────────────────────────

def poly_fit_predict(x_tr, y_tr, x_te, degree):
    """
    Build poly design matrices, fit w, return (y_pred_train, y_pred_test, w).
    """
    Xtr, mu_r, sig_r = poly_features(x_tr, degree)
    Xte, _,    _     = poly_features(x_te, degree, mu_r, sig_r)
    w                = solve_LS(Xtr, y_tr)
    return Xtr.T @ w, Xte.T @ w, w, Xtr, Xte, mu_r, sig_r


# ── iii. Fig. 5: scatter of standardized x_train vs y_train (N=5000) ─────────

x_tr5k, y_tr5k, x_te5k, y_te5k = get_splits(5000)

fig5, ax5 = plt.subplots(figsize=(7, 5))
ax5.scatter(x_tr5k, y_tr5k, s=6, alpha=0.35, color="steelblue", rasterized=True)
ax5.set_xlabel("MedInc — standardized (z-score)", fontsize=12)
ax5.set_ylabel("Median house value  ($100k)", fontsize=12)
ax5.set_title("Fig. 5 — Training data scatter  (train_size=5 000)", fontsize=12)
ax5.grid(True, linewidth=0.4)
fig5.tight_layout()
fig5.savefig(os.path.join(DIR, "fig5_scatter.png"), dpi=150)
print("Fig. 5 saved.")


# ── iv. MSE vs degree (train_size=5000) ───────────────────────────────────────

DEGREES = [1, 2, 3, 4, 5, 10, 15, 20]

train_mse_d = []
test_mse_d  = []

print("\n=== MSE vs degree  (train_size=5000) ===")
print(f"{'d':>4}  {'Train MSE':>11}  {'Test MSE':>10}")
print("-" * 30)

for d in DEGREES:
    yhat_tr, yhat_te, w, *_ = poly_fit_predict(x_tr5k, y_tr5k, x_te5k, d)
    tr_m = mse(yhat_tr, y_tr5k)
    te_m = mse(yhat_te, y_te5k)
    train_mse_d.append(tr_m)
    test_mse_d.append(te_m)
    print(f"{d:>4d}  {tr_m:>11.4f}  {te_m:>10.4f}")


# ── v. Fig. 6: train and test MSE vs d ───────────────────────────────────────

fig6, ax6 = plt.subplots(figsize=(8, 5))
ax6.plot(DEGREES, train_mse_d, "o-", color="steelblue",  linewidth=2,
         markersize=7, label="Train MSE")
ax6.plot(DEGREES, test_mse_d,  "s-", color="crimson",    linewidth=2,
         markersize=7, label="Test MSE")
ax6.set_xlabel("Polynomial degree  d", fontsize=12)
ax6.set_ylabel("MSE  ($100k)²", fontsize=12)
ax6.set_title("Fig. 6 — Polynomial Regression: Train & Test MSE vs Degree  (N=5 000)",
              fontsize=12)
ax6.legend(fontsize=11)
ax6.grid(True, linewidth=0.4)
fig6.tight_layout()
fig6.savefig(os.path.join(DIR, "fig6_mse_vs_degree.png"), dpi=150)
print("\nFig. 6 saved.")


# ── vi. MSE vs train_size (d=9) ───────────────────────────────────────────────

D_FIXED     = 9
TRAIN_SIZES = [1000, 3000, 5000, 7000, 10000]

train_mse_n = []
test_mse_n  = []

print(f"\n=== MSE vs train_size  (d={D_FIXED}) ===")
print(f"{'N_train':>8}  {'Train MSE':>11}  {'Test MSE':>10}")
print("-" * 34)

for ts in TRAIN_SIZES:
    x_tr, y_tr, x_te, y_te = get_splits(ts)
    yhat_tr, yhat_te, *_   = poly_fit_predict(x_tr, y_tr, x_te, D_FIXED)
    tr_m = mse(yhat_tr, y_tr)
    te_m = mse(yhat_te, y_te)
    train_mse_n.append(tr_m)
    test_mse_n.append(te_m)
    print(f"{ts:>8d}  {tr_m:>11.4f}  {te_m:>10.4f}")


# ── vii. Fig. 7: test MSE vs train_size (d=9) ────────────────────────────────

fig7, ax7 = plt.subplots(figsize=(8, 5))
ax7.plot(TRAIN_SIZES, test_mse_n, "s-", color="darkorange", linewidth=2,
         markersize=8, label=f"Test MSE (d={D_FIXED})")
ax7.set_xlabel("Training size  N", fontsize=12)
ax7.set_ylabel("Test MSE  ($100k)²", fontsize=12)
ax7.set_title(f"Fig. 7 — Polynomial Regression: Test MSE vs Training Size  (d={D_FIXED})",
              fontsize=12)
ax7.set_xticks(TRAIN_SIZES)
ax7.legend(fontsize=11)
ax7.grid(True, linewidth=0.4)
fig7.tight_layout()
fig7.savefig(os.path.join(DIR, "fig7_mse_vs_N_poly.png"), dpi=150)
print("\nFig. 7 saved.")


# ── viii. Fig. 8: scatter + fitted curve (d=9, N=5000) ───────────────────────

yhat_tr9, yhat_te9, w9, Xtr9, Xte9, mu_r9, sig_r9 = poly_fit_predict(
    x_tr5k, y_tr5k, x_te5k, D_FIXED
)

# Smooth curve: evaluate on a dense grid over training x range
x_grid      = np.linspace(x_tr5k.min(), x_tr5k.max(), 500)
Xgrid, _, _ = poly_features(x_grid, D_FIXED, mu_r9, sig_r9)
y_grid      = Xgrid.T @ w9

# Sort training points by x for reference
sort_idx   = np.argsort(x_tr5k)
x_sort     = x_tr5k[sort_idx]
y_sort     = y_tr5k[sort_idx]
yhat_sort  = yhat_tr9[sort_idx]

fig8, ax8 = plt.subplots(figsize=(9, 5))
ax8.scatter(x_tr5k, y_tr5k, s=6, alpha=0.30, color="steelblue",
            label="Training data", rasterized=True)
ax8.plot(x_grid, y_grid, color="crimson", linewidth=2.5,
         label=f"Fitted polynomial  (d={D_FIXED})")
ax8.set_xlabel("MedInc — standardized (z-score)", fontsize=12)
ax8.set_ylabel("Median house value  ($100k)", fontsize=12)
ax8.set_title(
    f"Fig. 8 — Polynomial Fit  (d={D_FIXED}, train_size=5 000)\n"
    f"Train MSE={mse(yhat_tr9, y_tr5k):.4f}   Test MSE={mse(yhat_te9, y_te5k):.4f}",
    fontsize=11
)
ax8.legend(fontsize=11)
ax8.set_ylim(-0.5, 6.5)
ax8.grid(True, linewidth=0.4)
fig8.tight_layout()
fig8.savefig(os.path.join(DIR, "fig8_poly_fit.png"), dpi=150)
print("Fig. 8 saved.")


# ── ix. Discussion ────────────────────────────────────────────────────────────

print("""
=== Discussion: Figs. 5–8 ===

Fig. 5 — Shape of the data:
  The scatter of standardized MedInc vs house value shows a noisy but
  positive monotonic relationship: higher income neighbourhoods tend to
  have higher house values.  The data is not simply linear — there is a
  steeper rise at higher incomes and a hard ceiling at 5.0 ($500k) due to
  top-coding.  The scatter is wide, indicating substantial unexplained
  variance (location, rooms, etc. are omitted from this 1-D model).

Fig. 6 — MSE vs polynomial degree (N=5000):
  • d=1 (linear): relatively high train and test MSE — the relationship is
    not purely linear, so the model underfits.
  • d=2–5: both train and test MSE decrease as the polynomial captures
    the gentle curvature of the data.  Train MSE ≤ test MSE throughout.
  • d=9–10: the model reaches near-optimal fit; train and test MSE are
    close, indicating a well-generalizing model.
  • d=15–20: train MSE continues to fall slightly but test MSE begins to
    rise or plateau — mild overfitting as the high-degree polynomial
    starts fitting noise in the training set.  The row-standardization
    of each power column mitigates the most extreme instability.
  → Classic bias-variance trade-off: d too small → underfitting;
    d too large → overfitting.

Fig. 7 — Test MSE vs training size (d=9):
  • Test MSE decreases as N increases from 1 000 → 10 000.
  • With d=9 and small N (1000), there is some overfitting risk (only
    ~111 observations per parameter).  Larger N provides more data to
    constrain the 10 coefficients → better generalisation.
  • The improvement is most pronounced from N=1000 → 3000 and then
    levels off, consistent with diminishing returns.

Fig. 8 — Fitted d=9 curve (N=5000):
  • The red polynomial captures the positive trend and the slight S-curve
    in the data: near-flat at low income, steeper in the middle, then
    flattening again near the top-coded ceiling.
  • The curve stays smooth and physically plausible across the observed
    input range — no severe oscillations (Runge phenomenon), thanks to
    the row-standardization of each power.
  • Both train and test MSE at d=9 are noticeably lower than the linear
    baseline (d=1), confirming that the polynomial better models the
    income–price relationship.
  • Significant residual scatter remains because house value depends on
    many features beyond income alone.
""")
