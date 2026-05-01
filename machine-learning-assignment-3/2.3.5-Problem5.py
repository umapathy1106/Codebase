"""
Problem 5 — Auto-MPG: Standardization, Pearson Correlation, and EDA Plots.

Matrices (column-major convention matching the problem statement):
  TRI ∈ R^{7 × ntrain}  — standardized training features
  TRO ∈ R^{ntrain × 1}  — training targets (mpg, NOT standardized)
  TEI ∈ R^{7 × ntest}   — standardized test features
  TEO ∈ R^{ntest × 1}   — test targets (mpg, NOT standardized)
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

DIR = os.path.dirname(os.path.abspath(__file__))

# ── Reproducibility ───────────────────────────────────────────────────────────
np.random.seed(42)

# ── Load Auto-MPG dataset ─────────────────────────────────────────────────────
url = ("https://archive.ics.uci.edu/ml/machine-learning-databases/"
       "auto-mpg/auto-mpg.data")
col_names = ["mpg", "cylinders", "displacement", "horsepower",
             "weight", "acceleration", "model_year", "origin", "car_name"]
feature_cols = ["cylinders", "displacement", "horsepower",
                "weight", "acceleration", "model_year", "origin"]

df = pd.read_csv(url, sep=r"\s+", names=col_names, na_values="?")
df = df.dropna(subset=["horsepower"]).reset_index(drop=True)

X_all = df[feature_cols].values.astype(float)   # (N, 7)
y_all = df["mpg"].values.astype(float)           # (N,)
N     = X_all.shape[0]                           # 392

# ── Global shuffle & split ────────────────────────────────────────────────────
indices   = np.random.permutation(N)
ntrain    = 300
ntest     = 92
train_idx = indices[:ntrain]
test_idx  = indices[ntrain:ntrain + ntest]

X_tr_raw = X_all[train_idx]   # (300, 7)
y_tr     = y_all[train_idx]   # (300,)
X_te_raw = X_all[test_idx]    # (92, 7)
y_te     = y_all[test_idx]    # (92,)


# ── i. Z-score standardization (row = feature) ───────────────────────────────
# Per-feature statistics from training set
mu_tr    = X_tr_raw.mean(axis=0)   # (7,)
sigma_tr = X_tr_raw.std(axis=0)    # (7,)

X_tr_z = (X_tr_raw - mu_tr) / sigma_tr   # (300, 7)
X_te_z = (X_te_raw - mu_tr) / sigma_tr   # (92,  7)

# Problem notation: features as rows
TRI = X_tr_z.T          # (7, 300)   — standardized training features
TRO = y_tr.reshape(-1, 1)   # (300, 1)  — training targets (raw mpg)
TEI = X_te_z.T          # (7, 92)    — standardized test features
TEO = y_te.reshape(-1, 1)   # (92, 1)   — test targets   (raw mpg)

print("=== Matrix shapes ===")
print(f"  TRI: {TRI.shape}  TRO: {TRO.shape}")
print(f"  TEI: {TEI.shape}  TEO: {TEO.shape}")

print("\n=== Feature standardization stats (training) ===")
print(f"{'Feature':>14}  {'µ':>10}  {'σ':>10}")
print("-" * 38)
for i, c in enumerate(feature_cols):
    print(f"{c:>14}  {mu_tr[i]:>10.4f}  {sigma_tr[i]:>10.4f}")


# ── ii. 8×8 Pearson correlation matrix C ─────────────────────────────────────

# Stack all 8 variables (7 raw features + mpg) column-wise using training data
# Use raw features for correlation (correlation is scale-invariant)
labels  = feature_cols + ["mpg"]
Z_corr  = np.column_stack([X_tr_raw, y_tr])      # (300, 8)

# Pearson correlation
def pearson_corr_matrix(Z):
    """Compute (n_vars × n_vars) Pearson correlation matrix from (n, p) data."""
    Z_c  = Z - Z.mean(axis=0)
    cov  = Z_c.T @ Z_c / (Z.shape[0] - 1)
    std  = np.sqrt(np.diag(cov))
    corr = cov / np.outer(std, std)
    return corr

C_raw = pearson_corr_matrix(Z_corr)          # (8, 8)  — signed
C_abs = np.abs(C_raw)                         # (8, 8)  — absolute values

print("\n=== 8×8 Absolute Pearson Correlation Matrix C (training set) ===")
header = f"{'':>14}" + "".join(f"{l:>14}" for l in labels)
print(header)
print("-" * (14 * 9))
for i, li in enumerate(labels):
    row = f"{li:>14}" + "".join(f"{C_abs[i, j]:>14.4f}" for j in range(8))
    print(row)

# Which input correlates most strongly with y (mpg)?
mpg_idx  = 7
feat_corr = C_abs[mpg_idx, :7]   # absolute correlations of x1..x7 with mpg
best_feat_idx = int(np.argmax(feat_corr))
print(f"\n  Feature correlating most strongly with mpg:")
print(f"  → x{best_feat_idx+1} ({feature_cols[best_feat_idx]}):  |r| = {feat_corr[best_feat_idx]:.4f}")
print("\n  All |r| with mpg:")
for i, (fc, r) in enumerate(zip(feature_cols, feat_corr)):
    print(f"    x{i+1} ({fc:>14s}): {r:.4f}")


# ── iii. Fig. 9: TRO vs TRI[0,:] (cylinders) ─────────────────────────────────

fig9, ax9 = plt.subplots(figsize=(7, 5))
ax9.scatter(TRI[0], TRO[:, 0], s=20, alpha=0.6, color="steelblue",
            edgecolors="none")
ax9.set_xlabel("Cylinders  (standardized)", fontsize=12)
ax9.set_ylabel("MPG", fontsize=12)
ax9.set_title("Fig. 9 — MPG vs Cylinders  (training set, N=300)", fontsize=12)
ax9.grid(True, linewidth=0.4)
fig9.tight_layout()
fig9.savefig(os.path.join(DIR, "fig9_mpg_vs_cylinders.png"), dpi=150)
print("\nFig. 9 saved.")


# ── iv. Fig. 10: TRO vs TRI[3,:] (weight) ────────────────────────────────────

fig10, ax10 = plt.subplots(figsize=(7, 5))
ax10.scatter(TRI[3], TRO[:, 0], s=20, alpha=0.6, color="darkorange",
             edgecolors="none")
ax10.set_xlabel("Weight  (standardized)", fontsize=12)
ax10.set_ylabel("MPG", fontsize=12)
ax10.set_title("Fig. 10 — MPG vs Weight  (training set, N=300)", fontsize=12)
ax10.grid(True, linewidth=0.4)
fig10.tight_layout()
fig10.savefig(os.path.join(DIR, "fig10_mpg_vs_weight.png"), dpi=150)
print("Fig. 10 saved.")


# ── v. Fig. 11: 3×3 grid — all 7 features vs mpg ────────────────────────────
# Discrete features: cylinders (row 0), model_year (row 5), origin (row 6)
# Apply jitter for discrete features.

rng     = np.random.default_rng(0)   # separate RNG for jitter only
DISCRETE = {0, 5, 6}                 # row indices that are discrete

fig11, axes = plt.subplots(3, 3, figsize=(14, 11))
axes = axes.ravel()

colors = ["steelblue", "darkorange", "seagreen", "crimson",
          "mediumpurple", "saddlebrown", "teal"]

for i in range(7):
    ax = axes[i]
    xvals = TRI[i].copy()
    if i in DISCRETE:
        xvals += rng.uniform(-0.05, 0.05, size=len(xvals))   # small jitter
    ax.scatter(xvals, TRO[:, 0], s=12, alpha=0.55,
               color=colors[i], edgecolors="none")
    xlabel = f"x{i+1}: {feature_cols[i]}" + (" (jittered)" if i in DISCRETE else "")
    xlabel += "  [std]"
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel("MPG", fontsize=9)
    ax.set_title(f"|r|={feat_corr[i]:.3f}", fontsize=9)
    ax.grid(True, linewidth=0.3)

# Hide unused subplots (positions 7 and 8)
for j in range(7, 9):
    axes[j].set_visible(False)

fig11.suptitle("Fig. 11 — All 7 Features vs MPG  (training set, N=300)\n"
               "Discrete features: cylinders, model_year, origin (jittered)",
               fontsize=12)
fig11.tight_layout()
fig11.savefig(os.path.join(DIR, "fig11_all_features_vs_mpg.png"), dpi=150)
print("Fig. 11 (3×3 grid) saved.")


# ── Correlation heatmap (bonus — useful for report) ──────────────────────────

fig_c, ax_c = plt.subplots(figsize=(8, 7))
im = ax_c.imshow(C_abs, cmap="YlOrRd", vmin=0, vmax=1)
plt.colorbar(im, ax=ax_c, label="|Pearson r|")
ax_c.set_xticks(range(8));  ax_c.set_xticklabels(labels, rotation=45, ha="right")
ax_c.set_yticks(range(8));  ax_c.set_yticklabels(labels)
for i in range(8):
    for j in range(8):
        ax_c.text(j, i, f"{C_abs[i,j]:.2f}", ha="center", va="center",
                  fontsize=8, color="black" if C_abs[i,j] < 0.7 else "white")
ax_c.set_title("Absolute Pearson Correlation Matrix  (training set)", fontsize=12)
fig_c.tight_layout()
fig_c.savefig(os.path.join(DIR, "fig_corr_heatmap.png"), dpi=150)
print("Correlation heatmap saved.")
