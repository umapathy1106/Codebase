import numpy as np
from sklearn.datasets import fetch_california_housing

# Reproducibility
# Note: the random seed only affects the train/test split, not the original dataset order.
np.random.seed(42)

# Load dataset
housing = fetch_california_housing()
X_all = housing.data          # shape (20640, 8)
y_all = housing.target        # shape (20640,)  — units: $100k

# Extract only the MedInc feature (column 0)
X_all = X_all[:, 0].reshape(-1, 1)   # shape (20640, 1)

# Global shuffle of indices
N = X_all.shape[0]  # 20640
indices = np.random.permutation(N)

# Split indices
train_size = 10000
train_idx = indices[0:train_size]          # 10 000 training samples
test_idx  = indices[10000:15000]           # 5 000 fixed test samples
# indices[15000:] are ignored

X_train_raw = X_all[train_idx]   # (10000, 1)
y_train      = y_all[train_idx]  # (10000,)

X_test_raw  = X_all[test_idx]    # (5000, 1)
y_test       = y_all[test_idx]   # (5000,)

# Standardize features using training mean/std only
# y is NOT standardized (raw MSE units)
train_mean = X_train_raw.mean(axis=0)   # shape (1,)
train_std  = X_train_raw.std(axis=0)    # shape (1,)

X_train = (X_train_raw - train_mean) / train_std   # (10000, 1)
X_test  = (X_test_raw  - train_mean) / train_std   # (5000,  1)

# Convenience: flatten to 1-D vectors for scalar-x regression
x_train = X_train.ravel()   # (10000,)
x_test  = X_test.ravel()    # (5000,)

if __name__ == "__main__":
    print("=== Dataset A: California Housing ===")
    print(f"Total samples  : {N}")
    print(f"Train size     : {x_train.shape[0]}")
    print(f"Test size      : {x_test.shape[0]}")
    print(f"Feature        : MedInc (standardized)")
    print(f"  train mean (raw): {train_mean[0]:.4f}, std (raw): {train_std[0]:.4f}")
    print(f"  x_train range  : [{x_train.min():.3f}, {x_train.max():.3f}]")
    print(f"  x_test  range  : [{x_test.min():.3f},  {x_test.max():.3f}]")
    print(f"Target (y) — NOT standardized:")
    print(f"  y_train range  : [{y_train.min():.3f}, {y_train.max():.3f}]  (units: $100k)")
    print(f"  y_test  range  : [{y_test.min():.3f}, {y_test.max():.3f}]  (units: $100k)")

    # Print top 20 training samples
    print("\n--- Top 20 Training Samples ---")
    print(f"{'#':<5} {'MedInc (std)':>15} {'House Value ($100k)':>20}")
    print("-" * 43)
    for i in range(20):
        print(f"{i+1:<5} {x_train[i]:>15.4f} {y_train[i]:>20.4f}")

    # Print top 20 test samples
    print("\n--- Top 20 Test Samples ---")
    print(f"{'#':<5} {'MedInc (std)':>15} {'House Value ($100k)':>20}")
    print("-" * 43)
    for i in range(20):
        print(f"{i+1:<5} {x_test[i]:>15.4f} {y_test[i]:>20.4f}")

    # Save full dataset to CSV
    import os
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "california_housing_full.csv")
    header = "#,Split,MedInc (std),House Value ($100k)"
    rows = []
    for i in range(len(x_train)):
        rows.append(f"{i+1},train,{x_train[i]:.6f},{y_train[i]:.6f}")
    for i in range(len(x_test)):
        rows.append(f"{len(x_train)+i+1},test,{x_test[i]:.6f},{y_test[i]:.6f}")

    with open(output_path, "w") as f:
        f.write(header + "\n")
        f.write("\n".join(rows) + "\n")

    print(f"\nFull dataset saved to: {output_path}")
    print(f"  Rows written: {len(rows)} ({len(x_train)} train + {len(x_test)} test)")
