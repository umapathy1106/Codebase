"""
Shared utilities for Part I: Parametric and Non-Parametric Regression.

Conventions enforced here:
  - numpy.random.seed(42) must be set by the caller before any random ops.
  - Features are Z-score standardized using training statistics only.
  - Target y is NEVER standardized; all MSE is in original y-units.
  - MSE = (1/n_test) * sum((y_hat - y)^2)
  - No scikit-learn for core algorithms (only allowed for dataset loading).
"""

import numpy as np
from sklearn.datasets import fetch_california_housing as _fetch_housing


# ---------------------------------------------------------------------------
# Z-score standardization (population std, ddof=0)
# ---------------------------------------------------------------------------

def standardize(X_train, X_test):
    """
    Standardize features using training mean and std (ddof=0).

    Parameters
    ----------
    X_train : ndarray, shape (n_train, d) or (n_train,)
    X_test  : ndarray, shape (n_test,  d) or (n_test,)

    Returns
    -------
    X_train_z, X_test_z : standardized arrays (same shape as inputs)
    mu, sigma           : training mean and std (shape (d,) or scalar)
    """
    mu    = X_train.mean(axis=0)
    sigma = X_train.std(axis=0)
    X_train_z = (X_train - mu) / sigma
    X_test_z  = (X_test  - mu) / sigma
    return X_train_z, X_test_z, mu, sigma


# ---------------------------------------------------------------------------
# MSE
# ---------------------------------------------------------------------------

def mse(y_pred, y_true):
    """
    Mean Squared Error in original y-units.

    MSE = (1/n) * sum((y_pred - y_true)^2)
    """
    y_pred = np.asarray(y_pred)
    y_true = np.asarray(y_true)
    return np.mean((y_pred - y_true) ** 2)


# ---------------------------------------------------------------------------
# Dataset A: California Housing
# ---------------------------------------------------------------------------

def load_california_housing(train_size=10000):
    """
    Load and preprocess Dataset A: California Housing.

    Parameters
    ----------
    train_size : int, <= 10000
        Number of training samples. Uses indices [0:train_size].
        Test set is always fixed at indices [10000:15000] (size 5000).

    Returns
    -------
    x_train : ndarray (train_size,)   — standardized MedInc (scalar feature)
    y_train : ndarray (train_size,)   — median house value in $100k
    x_test  : ndarray (5000,)         — standardized MedInc (same transform)
    y_test  : ndarray (5000,)         — median house value in $100k
    mu      : float                   — training mean of raw MedInc
    sigma   : float                   — training std  of raw MedInc
    """
    assert 1 <= train_size <= 10000, "train_size must be in [1, 10000]"

    housing = _fetch_housing()
    X_all   = housing.data[:, 0]          # MedInc only, shape (20640,)
    y_all   = housing.target              # shape (20640,)

    # Global shuffle — caller must have set np.random.seed(42) already
    N       = X_all.shape[0]             # 20640
    indices = np.random.permutation(N)

    train_idx = indices[0:train_size]
    test_idx  = indices[10000:15000]

    X_train_raw = X_all[train_idx]
    y_train     = y_all[train_idx]
    X_test_raw  = X_all[test_idx]
    y_test      = y_all[test_idx]

    # Standardize features only
    x_train, x_test, mu, sigma = standardize(X_train_raw, X_test_raw)

    return x_train, y_train, x_test, y_test, mu, sigma


# ---------------------------------------------------------------------------
# Dataset B: Auto-MPG
# ---------------------------------------------------------------------------

def load_auto_mpg(ntrain=300):
    """
    Load and preprocess Dataset B: Auto-MPG.

    Parameters
    ----------
    ntrain : int
        Number of training samples (default 300). Test = next 92 indices.

    Returns
    -------
    X_train : ndarray (ntrain, 7)  — standardized features
    y_train : ndarray (ntrain,)    — mpg
    X_test  : ndarray (92, 7)      — standardized features
    y_test  : ndarray (92,)        — mpg
    mu      : ndarray (7,)         — training feature means
    sigma   : ndarray (7,)         — training feature stds
    feature_names : list[str]
    """
    import pandas as pd

    url = ("https://archive.ics.uci.edu/ml/machine-learning-databases/"
           "auto-mpg/auto-mpg.data")
    col_names = ["mpg", "cylinders", "displacement", "horsepower",
                 "weight", "acceleration", "model_year", "origin", "car_name"]
    feature_cols = ["cylinders", "displacement", "horsepower",
                    "weight", "acceleration", "model_year", "origin"]

    df = pd.read_csv(url, sep=r"\s+", names=col_names, na_values="?")
    df = df.dropna(subset=["horsepower"]).reset_index(drop=True)

    X_all = df[feature_cols].values.astype(float)
    y_all = df["mpg"].values.astype(float)

    N       = X_all.shape[0]
    indices = np.random.permutation(N)

    train_idx = indices[0:ntrain]
    test_idx  = indices[ntrain:ntrain + 92]

    X_train_raw = X_all[train_idx]
    y_train     = y_all[train_idx]
    X_test_raw  = X_all[test_idx]
    y_test      = y_all[test_idx]

    X_train, X_test, mu, sigma = standardize(X_train_raw, X_test_raw)

    return X_train, y_train, X_test, y_test, mu, sigma, feature_cols


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    np.random.seed(42)

    print("=== utils.py self-test ===\n")

    # MSE sanity check
    assert mse(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0])) == 0.0
    assert abs(mse(np.array([0.0]), np.array([1.0])) - 1.0) < 1e-12
    print("MSE: OK")

    # California Housing
    x_tr, y_tr, x_te, y_te, mu, sigma = load_california_housing(train_size=10000)
    assert x_tr.shape == (10000,), f"Expected (10000,), got {x_tr.shape}"
    assert x_te.shape == (5000,),  f"Expected (5000,),  got {x_te.shape}"
    assert abs(x_tr.mean()) < 1e-10, "x_train should have ~zero mean"
    assert abs(x_tr.std()  - 1.0) < 1e-10, "x_train should have ~unit std"
    print(f"California Housing: train={x_tr.shape}, test={x_te.shape}  OK")
    print(f"  MedInc: mu={mu:.4f}, sigma={sigma:.4f}")

    # Auto-MPG
    np.random.seed(42)
    X_tr, y_tr2, X_te, y_te2, mu2, sigma2, fnames = load_auto_mpg(ntrain=300)
    assert X_tr.shape == (300, 7), f"Expected (300,7), got {X_tr.shape}"
    assert X_te.shape == (92,  7), f"Expected (92,7),  got {X_te.shape}"
    print(f"Auto-MPG:           train={X_tr.shape}, test={X_te.shape}  OK")
    print(f"  Features: {fnames}")

    print("\nAll checks passed.")
