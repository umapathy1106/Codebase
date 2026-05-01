"""
Part II — Classification, SGD, Cross-Validation
3.1 Dataset: Wine (UCI)

178 samples, 13 features, 3 classes.
All random operations use numpy.random.seed(42).

Conventions
-----------
- Features are loaded from the UCI repository.
- Class labels are mapped to 0-based integers: {1,2,3} → {0,1,2}.
- Z-score standardisation (ddof=0) with training statistics only.
- Train/test split: first 142 (≈80%) train, last 36 (≈20%) test
  after a seed-42 random permutation of all 178 samples.
- y is NOT standardised.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

DIR = os.path.dirname(os.path.abspath(__file__))

np.random.seed(42)

# ── Load from UCI ─────────────────────────────────────────────────────────────
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
print(f"Loaded Wine dataset: {df.shape[0]} samples, {df.shape[1]-1} features, "
      f"{df['class'].nunique()} classes")
print(f"Class distribution: {df['class'].value_counts().sort_index().to_dict()}")

# ── Extract arrays (0-based class labels) ────────────────────────────────────
X_all = df[feature_names].values.astype(float)   # (178, 13)
y_all = df["class"].values.astype(int) - 1        # {1,2,3} → {0,1,2}
N     = X_all.shape[0]                            # 178

print(f"\nFull dataset:  X shape = {X_all.shape},  y shape = {y_all.shape}")
print(f"Classes after remapping: {np.unique(y_all)}")

# ── Train / Test split ────────────────────────────────────────────────────────
# 80/20 split: 142 train, 36 test
indices   = np.random.permutation(N)
ntrain    = 142
ntest     = N - ntrain   # 36
train_idx = indices[:ntrain]
test_idx  = indices[ntrain:]

X_tr_raw = X_all[train_idx]   # (142, 13)
y_tr     = y_all[train_idx]   # (142,)
X_te_raw = X_all[test_idx]    # (36,  13)
y_te     = y_all[test_idx]    # (36,)

print(f"\nSplit:  train = {X_tr_raw.shape},  test = {X_te_raw.shape}")
print(f"Train class distribution: { {c: int((y_tr==c).sum()) for c in range(3)} }")
print(f"Test  class distribution: { {c: int((y_te==c).sum()) for c in range(3)} }")

# ── Z-score standardisation (training stats, ddof=0) ─────────────────────────
mu_feat    = X_tr_raw.mean(axis=0)    # (13,)
sigma_feat = X_tr_raw.std(axis=0)     # (13,)

X_tr = (X_tr_raw - mu_feat) / sigma_feat   # (142, 13)
X_te = (X_te_raw - mu_feat) / sigma_feat   # (36,  13)

print(f"\nAfter standardisation:")
print(f"  Train mean (should be ~0):  {X_tr.mean(axis=0).round(4)}")
print(f"  Train std  (should be ~1):  {X_tr.std(axis=0).round(4)}")

# ── Summary table ─────────────────────────────────────────────────────────────
print("\n=== Top 10 rows (raw) ===")
print(df.head(10).to_string(index=False))

print("\n=== Feature statistics (training set, raw) ===")
stats = pd.DataFrame({
    "feature" : feature_names,
    "min"     : X_tr_raw.min(axis=0),
    "max"     : X_tr_raw.max(axis=0),
    "mean"    : X_tr_raw.mean(axis=0),
    "std"     : X_tr_raw.std(axis=0),
})
print(stats.to_string(index=False))

# ── Export CSV ────────────────────────────────────────────────────────────────
out_df = pd.DataFrame(X_all, columns=feature_names)
out_df.insert(0, "class", y_all)   # 0-based labels
csv_path = os.path.join(DIR, "wine_full.csv")
out_df.to_csv(csv_path, index=False)
print(f"\nExported full dataset → {csv_path}")
print(f"  Rows: {len(out_df)},  Columns: {list(out_df.columns)}")
