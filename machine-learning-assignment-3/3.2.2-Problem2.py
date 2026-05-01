"""
Part II — Classification, SGD, Cross-Validation
3.2.2 Problem 2

Implements four utility functions used throughout Part II:
  ii. softmax(z)             — numerically stable softmax
  ii. kfold_split(n, K)      — K-fold cross-validation index generator
  ii. accuracy(y_true, y_pred) — classification accuracy
  ii. confusion_matrix(y_true, y_pred, n_classes) — confusion matrix
"""

import numpy as np

np.random.seed(42)


# ─────────────────────────────────────────────────────────────────────────────
# ii.  softmax
# ─────────────────────────────────────────────────────────────────────────────

def softmax(z):
    """
    Numerically stable softmax over a 1-D logit vector z (length C).

    Subtracts max(z) before exponentiation to prevent overflow while
    leaving the output probabilities unchanged:
        softmax(z)_k = exp(z_k - max(z)) / sum_j exp(z_j - max(z))

    Parameters
    ----------
    z : array-like, shape (C,)
        Raw logit scores for C classes.

    Returns
    -------
    p : ndarray, shape (C,)
        Probability vector; entries are positive and sum to 1.
    """
    z = np.asarray(z, dtype=float)
    e = np.exp(z - np.max(z))
    return e / e.sum()


# ─────────────────────────────────────────────────────────────────────────────
# ii.  kfold_split
# ─────────────────────────────────────────────────────────────────────────────

def kfold_split(n, K):
    """
    Generate train/validation index pairs for K-fold cross-validation.

    Splits the index range [0, n) into K roughly equal consecutive folds.
    Fold k uses indices[k] as validation; the remaining K-1 folds as training.

    Parameters
    ----------
    n : int
        Total number of samples.
    K : int
        Number of folds.

    Yields
    ------
    train_idx : ndarray of int
        Indices of training samples for this fold.
    val_idx   : ndarray of int
        Indices of validation samples for this fold.
    """
    indices = np.arange(n)
    fold_sizes = np.full(K, n // K, dtype=int)
    fold_sizes[: n % K] += 1          # distribute remainder to first folds

    starts = np.concatenate([[0], np.cumsum(fold_sizes)])

    for k in range(K):
        val_idx   = indices[starts[k] : starts[k + 1]]
        train_idx = np.concatenate([indices[: starts[k]],
                                    indices[starts[k + 1] :]])
        yield train_idx, val_idx


# ─────────────────────────────────────────────────────────────────────────────
# ii.  accuracy
# ─────────────────────────────────────────────────────────────────────────────

def accuracy(y_true, y_pred):
    """
    Classification accuracy: fraction of correctly predicted labels.

        accuracy = (number of correct predictions) / (total predictions)

    Parameters
    ----------
    y_true : array-like, shape (n,)   — ground-truth integer class labels.
    y_pred : array-like, shape (n,)   — predicted integer class labels.

    Returns
    -------
    float in [0, 1].
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.sum(y_true == y_pred) / len(y_true))


# ─────────────────────────────────────────────────────────────────────────────
# ii.  confusion_matrix
# ─────────────────────────────────────────────────────────────────────────────

def confusion_matrix(y_true, y_pred, n_classes):
    """
    Build a confusion matrix of shape (n_classes, n_classes).

    CM[i, j] = number of samples with true class i predicted as class j.
    Rows → true class; columns → predicted class.

    Parameters
    ----------
    y_true    : array-like, shape (n,) — ground-truth labels in {0,..,C-1}.
    y_pred    : array-like, shape (n,) — predicted labels in {0,..,C-1}.
    n_classes : int — number of classes C.

    Returns
    -------
    CM : ndarray, shape (n_classes, n_classes), dtype int.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    CM = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        CM[t, p] += 1
    return CM


# ─────────────────────────────────────────────────────────────────────────────
# Self-tests
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    print("=" * 60)
    print("Test 1 — softmax")
    print("=" * 60)
    z = np.array([2.0, 1.0, 0.1])
    p = softmax(z)
    print(f"  z = {z}")
    print(f"  softmax(z) = {p.round(6)}")
    print(f"  sum = {p.sum():.6f}  (should be 1)")
    assert abs(p.sum() - 1.0) < 1e-12, "softmax does not sum to 1"

    # Numerical stability: large logits
    z_large = np.array([1000.0, 999.0, 998.0])
    p_large = softmax(z_large)
    print(f"  softmax([1000,999,998]) = {p_large.round(6)}  (no overflow)")
    assert np.all(np.isfinite(p_large)), "softmax overflow / NaN detected"
    print("  PASS\n")

    print("=" * 60)
    print("Test 2 — kfold_split")
    print("=" * 60)
    n, K = 12, 4
    fold_sizes = []
    for fold_i, (tr_idx, va_idx) in enumerate(kfold_split(n, K)):
        fold_sizes.append(len(va_idx))
        all_idx = np.sort(np.concatenate([tr_idx, va_idx]))
        assert np.array_equal(all_idx, np.arange(n)), \
            f"Fold {fold_i}: train+val does not cover all indices"
        assert len(np.intersect1d(tr_idx, va_idx)) == 0, \
            f"Fold {fold_i}: train/val overlap"
        print(f"  Fold {fold_i}: train={len(tr_idx)}, val={len(va_idx)}, "
              f"val_idx={va_idx}")
    assert sum(fold_sizes) == n, "Fold sizes do not sum to n"

    # Uneven split: n=10, K=3 → folds of size 4,3,3
    folds_10 = [(tr, va) for tr, va in kfold_split(10, 3)]
    sizes_10 = [len(va) for _, va in folds_10]
    print(f"\n  n=10, K=3 fold sizes: {sizes_10}  (expected [4,3,3])")
    assert sizes_10 == [4, 3, 3], f"Unexpected fold sizes: {sizes_10}"
    print("  PASS\n")

    print("=" * 60)
    print("Test 3 — accuracy")
    print("=" * 60)
    y_true_t = np.array([0, 1, 2, 0, 1, 2])
    y_pred_t = np.array([0, 1, 2, 1, 1, 0])   # 4/6 correct
    acc = accuracy(y_true_t, y_pred_t)
    print(f"  y_true = {y_true_t}")
    print(f"  y_pred = {y_pred_t}")
    print(f"  accuracy = {acc:.4f}  (expected {4/6:.4f})")
    assert abs(acc - 4 / 6) < 1e-10, "accuracy mismatch"
    print("  PASS\n")

    print("=" * 60)
    print("Test 4 — confusion_matrix")
    print("=" * 60)
    CM = confusion_matrix(y_true_t, y_pred_t, n_classes=3)
    print(f"  y_true = {y_true_t}")
    print(f"  y_pred = {y_pred_t}")
    print("  Confusion matrix (rows=true, cols=pred):")
    print("  " + str(CM).replace("\n", "\n  "))
    # Class 0: true=2 samples → pred 0 once, pred 1 once  → CM[0,0]=1, CM[0,1]=1
    # Class 1: true=2 samples → pred 1 twice              → CM[1,1]=2
    # Class 2: true=2 samples → pred 2 once, pred 0 once  → CM[2,2]=1, CM[2,0]=1
    expected = np.array([[1, 1, 0],
                         [0, 2, 0],
                         [1, 0, 1]])
    assert np.array_equal(CM, expected), f"CM mismatch:\n{CM}\nvs\n{expected}"
    print(f"  Sum of CM = {CM.sum()}  (should equal n = {len(y_true_t)})")
    assert CM.sum() == len(y_true_t), "CM sum != n"
    print("  PASS\n")

    print("All tests passed.")
