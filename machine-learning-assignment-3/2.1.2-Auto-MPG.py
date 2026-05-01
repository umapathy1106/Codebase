import numpy as np
import pandas as pd

# Reproducibility
np.random.seed(42)

# Load dataset from UCI repository
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
col_names = ["mpg", "cylinders", "displacement", "horsepower",
             "weight", "acceleration", "model_year", "origin", "car_name"]

df = pd.read_csv(url, sep=r"\s+", names=col_names, na_values="?")

# Preprocessing: drop rows with missing horsepower
df = df.dropna(subset=["horsepower"]).reset_index(drop=True)

# Features (7) and target
feature_cols = ["cylinders", "displacement", "horsepower",
                "weight", "acceleration", "model_year", "origin"]
X_all = df[feature_cols].values.astype(float)   # shape (N, 7)
y_all = df["mpg"].values.astype(float)           # shape (N,)

N = X_all.shape[0]

# Global shuffle of indices
indices = np.random.permutation(N)

# Split
ntrain = 300
train_idx = indices[0:ntrain]           # 300 training samples
test_idx  = indices[ntrain:ntrain + 92] # 92 test samples

X_train_raw = X_all[train_idx]   # (300, 7)
y_train      = y_all[train_idx]  # (300,)

X_test_raw  = X_all[test_idx]    # (92, 7)
y_test       = y_all[test_idx]   # (92,)

# Standardize features using training mean/std only
# y is NOT standardized (raw MPG units)
train_mean = X_train_raw.mean(axis=0)   # shape (7,)
train_std  = X_train_raw.std(axis=0)    # shape (7,)

X_train = (X_train_raw - train_mean) / train_std   # (300, 7)
X_test  = (X_test_raw  - train_mean) / train_std   # (92,  7)

if __name__ == "__main__":
    import os

    print("=== Dataset B: Auto-MPG ===")
    print(f"Total samples (after dropping missing): {N}")
    print(f"Train size : {X_train.shape[0]}")
    print(f"Test size  : {X_test.shape[0]}")
    print(f"\nFeature standardization (training stats):")
    for i, col in enumerate(feature_cols):
        print(f"  x{i+1} ({col:>12s}): mean={train_mean[i]:8.3f}, std={train_std[i]:8.3f}")

    # Print top 20 training samples
    print("\n--- Top 20 Training Samples ---")
    header = f"{'#':<5}" + "".join(f"  x{i+1}({c[:4]})" for i, c in enumerate(feature_cols)) + "     mpg"
    print(header)
    print("-" * len(header))
    for i in range(20):
        row = f"{i+1:<5}" + "".join(f"{X_train[i, j]:>10.4f}" for j in range(7)) + f"{y_train[i]:>8.3f}"
        print(row)

    # Print top 20 test samples
    print("\n--- Top 20 Test Samples ---")
    print(header)
    print("-" * len(header))
    for i in range(20):
        row = f"{i+1:<5}" + "".join(f"{X_test[i, j]:>10.4f}" for j in range(7)) + f"{y_test[i]:>8.3f}"
        print(row)

    # Save full dataset to CSV
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "auto_mpg_full.csv")
    std_feature_names = [f"x{i+1}_{c}_std" for i, c in enumerate(feature_cols)]
    csv_header = "#,Split," + ",".join(std_feature_names) + ",mpg"

    rows = []
    for i in range(len(y_train)):
        vals = ",".join(f"{X_train[i, j]:.6f}" for j in range(7))
        rows.append(f"{i+1},train,{vals},{y_train[i]:.3f}")
    for i in range(len(y_test)):
        vals = ",".join(f"{X_test[i, j]:.6f}" for j in range(7))
        rows.append(f"{len(y_train)+i+1},test,{vals},{y_test[i]:.3f}")

    with open(output_path, "w") as f:
        f.write(csv_header + "\n")
        f.write("\n".join(rows) + "\n")

    print(f"\nFull dataset saved to: {output_path}")
    print(f"  Rows written: {len(rows)} ({len(y_train)} train + {len(y_test)} test)")
