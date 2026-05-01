"""
Problem 7 — Multivariate Polynomial Regression on Auto-MPG (total degree M ∈ {1,2,3}).

Tasks
-----
vii-1  Print explicit model terms for each M.
vii-2  Train on TRI/TRO; evaluate train & test MSE for each M.
vii-3  Fig. 14 — MSE vs M (train=red, test=blue); compare to best univariate
       result from Problem 6.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
from itertools import combinations_with_replacement
from math import comb

DIR = os.path.dirname(os.path.abspath(__file__))

# ── Reproducibility & data ────────────────────────────────────────────────────
np.random.seed(42)

url = ("https://archive.ics.uci.edu/ml/machine-learning-databases/"
       "auto-mpg/auto-mpg.data")
col_names    = ["mpg", "cylinders", "displacement", "horsepower",
                "weight", "acceleration", "model_year", "origin", "car_name"]
feature_cols = ["cylinders", "displacement", "horsepower",
                "weight", "acceleration", "model_year", "origin"]

df = pd.read_csv(url, sep=r"\s+", names=col_names, na_values="?")
df = df.dropna(subset=["horsepower"]).reset_index(drop=True)

X_all = df[feature_cols].values.astype(float)   # (392, 7)
y_all = df["mpg"].values.astype(float)           # (392,)
N     = X_all.shape[0]                           # 392

indices   = np.random.permutation(N)
ntrain, ntest = 300, 92
train_idx = indices[:ntrain]
test_idx  = indices[ntrain:ntrain + ntest]

X_tr_raw = X_all[train_idx]   # (300, 7)
y_tr     = y_all[train_idx]   # (300,)
X_te_raw = X_all[test_idx]    # (92,  7)
y_te     = y_all[test_idx]    # (92,)

# Per-feature z-score standardisation (training stats only)
mu_feat    = X_tr_raw.mean(axis=0)   # (7,)
sigma_feat = X_tr_raw.std(axis=0)    # (7,)

X_tr_z = (X_tr_raw - mu_feat) / sigma_feat   # (300, 7)
X_te_z = (X_te_raw - mu_feat) / sigma_feat   # (92,  7)

TRI = X_tr_z.T   # (7, 300) — column per sample
TRO = y_tr       # (300,)
TEI = X_te_z.T   # (7, 92)
TEO = y_te       # (92,)

d = TRI.shape[0]  # 7 features


# ── Helper: term label from multi-index ──────────────────────────────────────

def term_label(combo):
    """
    combo: tuple of 0-based feature indices (may repeat).
    Returns human-readable label, e.g. () → '1', (0,1) → 'x1·x2', (0,0) → 'x1²'.
    """
    if len(combo) == 0:
        return "1"
    from collections import Counter
    cnt = Counter(combo)
    parts = []
    for idx in sorted(cnt):
        exp = cnt[idx]
        if exp == 1:
            parts.append(f"x{idx+1}")
        elif exp == 2:
            parts.append(f"x{idx+1}\u00b2")  # ²
        elif exp == 3:
            parts.append(f"x{idx+1}\u00b3")  # ³
        else:
            parts.append(f"x{idx+1}^{exp}")
    return "\u00b7".join(parts)   # ·


# ── Helper: build multivariate poly design matrix ────────────────────────────

def multi_poly_features(X_z, degree, mu_rows=None, sigma_rows=None):
    """
    Build design matrix Phi (p, n) for multivariate input X_z (d, n).

    Includes all monomials of total degree 0..degree over the d features.
    Row 0 = bias (all-ones); rows 1.. = monomials in graded-lex order.
    Each non-bias row is row-standardised using training stats.

    Training mode: pass mu_rows=None → stats computed and returned.
    Test    mode:  pass precomputed mu_rows, sigma_rows → applied directly.

    Returns
    -------
    Phi       : (p, n) design matrix
    terms     : list of p tuples (multi-indices)
    mu_rows   : (p-1,)
    sigma_rows: (p-1,)
    """
    n = X_z.shape[1]
    raw_cols = [np.ones(n)]
    terms    = [()]

    for deg in range(1, degree + 1):
        for combo in combinations_with_replacement(range(d), deg):
            col = np.ones(n)
            for idx in combo:
                col = col * X_z[idx]
            raw_cols.append(col)
            terms.append(combo)

    Phi = np.vstack(raw_cols)   # (p, n)

    train_mode = (mu_rows is None)
    if train_mode:
        mu_rows    = Phi[1:].mean(axis=1)
        sigma_rows = Phi[1:].std(axis=1)
        sigma_rows = np.where(sigma_rows == 0, 1.0, sigma_rows)

    Phi[1:] = (Phi[1:] - mu_rows[:, None]) / sigma_rows[:, None]
    return Phi, terms, mu_rows, sigma_rows


# ── Helper: SVD least-squares ────────────────────────────────────────────────

def solve_LS(Phi, y):
    """Solve min ||Phi.T w - y||² via economy SVD; returns w (p,)."""
    U, s, Vt = np.linalg.svd(Phi, full_matrices=False)
    thresh    = 1e-10 * s[0] if s[0] > 0 else 1e-10
    s_inv     = np.where(s > thresh, 1.0 / s, 0.0)
    return U @ (s_inv * (Vt @ y))


def mse(yp, yt):
    return float(np.mean((yp - yt) ** 2))


# ─────────────────────────────────────────────────────────────────────────────
# Task vii-1  Explicit model terms for each M
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 70)
print("Task vii-1 — Explicit model terms")
print("=" * 70)

for M in [1, 2, 3]:
    # collect all terms
    all_terms = [()]
    for deg in range(1, M + 1):
        for combo in combinations_with_replacement(range(d), deg):
            all_terms.append(combo)
    p = len(all_terms)
    labels = [term_label(t) for t in all_terms]

    # expected count: sum_{k=0}^{M} C(d+k-1, k)  = C(d+M, M)
    p_expected = comb(d + M, M)
    assert p == p_expected, f"M={M}: got {p}, expected {p_expected}"

    print(f"\nM = {M}  →  {p} terms  [= C({d+M},{M})]")
    # Print at most first 40 + last 5 to keep output manageable
    if p <= 50:
        print("  " + ",  ".join(labels))
    else:
        shown = labels[:40]
        print("  " + ",  ".join(shown))
        print(f"  ... ({p - 45} terms omitted) ...")
        print("  " + ",  ".join(labels[-5:]))


# ─────────────────────────────────────────────────────────────────────────────
# Task vii-2  Train & evaluate for M ∈ {1, 2, 3}
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("Task vii-2 — Train / Test MSE for each M")
print("=" * 70)

DEGREES   = [1, 2, 3]
train_mse = {}
test_mse  = {}

for M in DEGREES:
    Phi_tr, terms, mu_r, sig_r = multi_poly_features(TRI, M)
    Phi_te, _,     _,    _     = multi_poly_features(TEI, M, mu_r, sig_r)

    w = solve_LS(Phi_tr, TRO)

    y_hat_tr = Phi_tr.T @ w
    y_hat_te = Phi_te.T @ w

    train_mse[M] = mse(y_hat_tr, TRO)
    test_mse[M]  = mse(y_hat_te, TEO)

    p = Phi_tr.shape[0]
    print(f"\n  M = {M}  (p = {p} parameters, n_train = {ntrain})")
    print(f"    Train MSE = {train_mse[M]:.4f} MPG²")
    print(f"    Test  MSE = {test_mse[M]:.4f}  MPG²")


# ─────────────────────────────────────────────────────────────────────────────
# Task vii-3  Fig. 14 — MSE vs M
# ─────────────────────────────────────────────────────────────────────────────

# Best univariate result from Problem 6 (weight, M=5, test MSE ≈ 13.07)
BEST_UNI_MSE = 13.07   # from Problem 6 table

fig14, ax14 = plt.subplots(figsize=(8, 5))

x_vals = list(DEGREES)
tr_vals = [train_mse[M] for M in DEGREES]
te_vals = [test_mse[M]  for M in DEGREES]

ax14.plot(x_vals, tr_vals, "ro-", linewidth=2, markersize=8,
          label="Train MSE")
ax14.plot(x_vals, te_vals, "bo-", linewidth=2, markersize=8,
          label="Test MSE")

# Horizontal reference: best univariate test MSE
ax14.axhline(BEST_UNI_MSE, color="green", linestyle="--", linewidth=1.5,
             label=f"Best Univariate Test MSE = {BEST_UNI_MSE:.2f} (Problem 6, weight M=5)")

# Annotate each point
for M in DEGREES:
    ax14.annotate(f"{train_mse[M]:.2f}", (M, train_mse[M]),
                  textcoords="offset points", xytext=(-18, 6),
                  fontsize=9, color="red")
    ax14.annotate(f"{test_mse[M]:.2f}", (M, test_mse[M]),
                  textcoords="offset points", xytext=(-18, -14),
                  fontsize=9, color="blue")

ax14.set_xticks(DEGREES)
ax14.set_xlabel("Polynomial Degree M  (total degree)", fontsize=12)
ax14.set_ylabel("MSE  (MPG²)", fontsize=12)
ax14.set_title("Fig. 14 — Multivariate Polynomial Regression: Train & Test MSE vs M\n"
               "(Auto-MPG, N=300, all 7 features)", fontsize=11)
ax14.legend(fontsize=9)
ax14.grid(True, linestyle="--", alpha=0.5)
ax14.set_xlim(0.7, 3.3)

fig14.tight_layout()
fig14_path = os.path.join(DIR, "fig14_multivariate_poly.png")
fig14.savefig(fig14_path, dpi=150)
plt.close(fig14)
print(f"\nFig. 14 saved → {fig14_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Discussion
# ─────────────────────────────────────────────────────────────────────────────

best_te_M  = min(DEGREES, key=lambda M: test_mse[M])
best_te_v  = test_mse[best_te_M]
improvement = 100.0 * (BEST_UNI_MSE - best_te_v) / BEST_UNI_MSE

print("\n" + "=" * 70)
print("Discussion")
print("=" * 70)
print(f"""
Model complexity
----------------
  M=1 : {comb(d+1,1):3d} parameters  – linear model over 7 features
  M=2 : {comb(d+2,2):3d} parameters  – adds all pairwise products & squares
  M=3 : {comb(d+3,3):3d} parameters  – adds all cubic monomials

Fit quality
-----------
  M=1 : train MSE = {train_mse[1]:.4f},  test MSE = {test_mse[1]:.4f}
  M=2 : train MSE = {train_mse[2]:.4f},  test MSE = {test_mse[2]:.4f}
  M=3 : train MSE = {train_mse[3]:.4f},  test MSE = {test_mse[3]:.4f}
  Best multivariate test MSE: M={best_te_M} → {best_te_v:.4f} MPG²

Comparison with Problem 6 (best univariate)
--------------------------------------------
  Univariate best (weight, M=5): {BEST_UNI_MSE:.2f} MPG²
  Multivariate best (M={best_te_M})       : {best_te_v:.4f} MPG²
  Improvement                  : {improvement:.1f}% lower test MSE

Overfitting & generalisation
-----------------------------
  M=3 has {comb(d+3,3)} parameters with only {ntrain} training samples
  (ratio ~{ntrain/comb(d+3,3):.1f}×). The training MSE continues to fall
  as M grows, but the test MSE behaviour reveals whether overfitting
  occurs. Joint information from all 7 features (especially the
  strongly correlated cluster: cylinders, displacement, horsepower,
  weight) allows the multivariate model to capture interactions
  that no single-feature model can, resulting in a substantially
  lower test error. The SVD pseudoinverse provides numerical
  stability and a minimum-norm solution even when columns are
  near-collinear.
""")
