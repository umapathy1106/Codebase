"""
Part II — Classification, SGD, Cross-Validation
3.2.1 Problem 1

Tasks
-----
i. Load the Wine dataset; extract feature matrix X and label vector y.
i. Filter to retain only the first 40 samples from each class (120 total).
i. Z-score normalise all features using population mean/std (ddof=0)
   computed on the 120-sample filtered set.
"""

import numpy as np
import pandas as pd
import os

np.random.seed(42)

DIR = os.path.dirname(os.path.abspath(__file__))

# ── i. Load Wine dataset ──────────────────────────────────────────────────────
url = ("https://archive.ics.uci.edu/ml/machine-learning-databases/"
       "wine/wine.data")

feature_names = [
    "alcohol", "malic_acid", "ash", "alcalinity_of_ash",
    "magnesium", "total_phenols", "flavanoids",
    "nonflavanoid_phenols", "proanthocyanins",
    "color_intensity", "hue",
    "od280_od315_of_diluted_wines", "proline"
]
col_names = ["class"] + feature_names

df = pd.read_csv(url, header=None, names=col_names)

# 0-based class labels: {1,2,3} → {0,1,2}
X_full = df[feature_names].values.astype(float)   # (178, 13)
y_full = df["class"].values.astype(int) - 1        # (178,)  ∈ {0,1,2}

print(f"Full dataset loaded:  X = {X_full.shape},  y = {y_full.shape}")
print(f"Class counts: { {c: int((y_full==c).sum()) for c in range(3)} }")

# ── i. Filter: first 40 samples from each class ───────────────────────────────
SAMPLES_PER_CLASS = 40
selected = []

for c in range(3):
    idx_c = np.where(y_full == c)[0]          # all indices for class c
    first40 = idx_c[:SAMPLES_PER_CLASS]        # first 40 in original order
    selected.append(first40)
    print(f"  Class {c}: using indices {first40[0]}..{first40[-1]}")

selected_idx = np.concatenate(selected)        # (120,)

X_filtered = X_full[selected_idx]             # (120, 13)
y_filtered = y_full[selected_idx]             # (120,)

print(f"\nFiltered dataset:  X = {X_filtered.shape},  y = {y_filtered.shape}")
print(f"Class distribution after filter: { {c: int((y_filtered==c).sum()) for c in range(3)} }")

# ── i. Z-score normalisation (population stats, ddof=0) ──────────────────────
mu    = X_filtered.mean(axis=0)   # (13,)  — population mean
sigma = X_filtered.std(axis=0)    # (13,)  — population std (ddof=0)

# Guard against zero std (should not occur for Wine, but good practice)
sigma_safe = np.where(sigma == 0, 1.0, sigma)

X_norm = (X_filtered - mu) / sigma_safe       # (120, 13)

print("\n=== Normalisation check ===")
print(f"  Mean of X_norm  (should be ~0): {X_norm.mean(axis=0).round(6)}")
print(f"  Std  of X_norm  (should be ~1): {X_norm.std(axis=0).round(6)}")

# ── Summary ────────────────────────────────────────────────────────────────────
print("\n=== Feature statistics (filtered, raw) ===")
stats = pd.DataFrame({
    "feature": feature_names,
    "mu":      mu.round(4),
    "sigma":   sigma.round(4),
    "min_raw": X_filtered.min(axis=0).round(4),
    "max_raw": X_filtered.max(axis=0).round(4),
})
print(stats.to_string(index=False))

print("\n=== First 5 rows of X_norm ===")
df_norm = pd.DataFrame(X_norm, columns=feature_names)
df_norm.insert(0, "class", y_filtered)
print(df_norm.head(5).to_string(index=False))

# Expose for downstream problems
print(f"\nReady for classification:  X_norm = {X_norm.shape},  y = {y_filtered.shape}")
