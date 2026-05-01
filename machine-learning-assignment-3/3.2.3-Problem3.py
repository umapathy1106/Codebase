"""
Part II — Classification, SGD, Cross-Validation
3.2.3 Problem 3

Mini-batch SGD with softmax cross-entropy on the Wine dataset (120 samples,
3 balanced classes, 13 features) using K-fold cross-validation.

Hyperparameters (fixed)
-----------------------
  Learning rate (η)  : 0.1
  Epochs per fold    : 500
  Mini-batch sizes   : 1, 36, 72
  K (folds)          : 10
  Weight init        : zeros
  Regularisation (λ) : 0
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import time
import os

np.random.seed(42)

DIR = os.path.dirname(os.path.abspath(__file__))

# ── Re-use utility functions from Problem 2 ───────────────────────────────────

def softmax_batch(logits):
    """Row-wise numerically stable softmax.  logits: (n, C) → (n, C)."""
    shifted = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(shifted)
    return e / e.sum(axis=1, keepdims=True)


def kfold_split(n, K):
    """Yield (train_idx, val_idx) for K consecutive folds."""
    indices   = np.arange(n)
    fold_sizes = np.full(K, n // K, dtype=int)
    fold_sizes[: n % K] += 1
    starts = np.concatenate([[0], np.cumsum(fold_sizes)])
    for k in range(K):
        val_idx   = indices[starts[k] : starts[k + 1]]
        train_idx = np.concatenate([indices[: starts[k]],
                                    indices[starts[k + 1] :]])
        yield train_idx, val_idx


def accuracy(y_true, y_pred):
    return float(np.sum(y_true == y_pred) / len(y_true))


def confusion_matrix(y_true, y_pred, n_classes):
    CM = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        CM[t, p] += 1
    return CM


# ── Load & prepare data (Problem 1 pipeline) ──────────────────────────────────

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
X_full = df[feature_names].values.astype(float)
y_full = df["class"].values.astype(int) - 1   # {1,2,3} → {0,1,2}

# First 40 samples per class
selected = [np.where(y_full == c)[0][:40] for c in range(3)]
sel_idx  = np.concatenate(selected)
X_filt   = X_full[sel_idx]   # (120, 13)
y_filt   = y_full[sel_idx]   # (120,)

# Z-score normalisation (population stats, ddof=0)
mu    = X_filt.mean(axis=0)
sigma = X_filt.std(axis=0)
X_norm = (X_filt - mu) / sigma   # (120, 13)

# Augment with bias column → (120, 14)
X_aug = np.hstack([X_norm, np.ones((X_norm.shape[0], 1))])

C, d = 3, X_aug.shape[1]   # 3 classes, 14 features (incl. bias)
N    = X_aug.shape[0]       # 120

print(f"Dataset ready: X_aug = {X_aug.shape}, y = {y_filt.shape}, C={C}")

# Pre-shuffle for balanced K-fold (class order would break fold balance)
preshuf = np.random.permutation(N)
X_shuf  = X_aug[preshuf]
y_shuf  = y_filt[preshuf]

# ── Hyperparameters ───────────────────────────────────────────────────────────

LR          = 0.1
EPOCHS      = 500
BATCH_SIZES = [1, 36, 72]
K_FOLDS     = 10

# ── SGD training function ─────────────────────────────────────────────────────

def train_fold(X_tr, y_tr, X_va, y_va, batch_size, epochs, lr):
    """
    Train a softmax linear classifier with mini-batch SGD.

    Gradient of softmax cross-entropy loss w.r.t. W (C × d):
        dL/dW = (1/B) * (P - one_hot(y)).T @ X_batch
    where P = softmax(X_batch @ W.T).

    Returns arrays of length `epochs`:
      tr_acc   — training accuracy after each epoch
      va_acc   — validation accuracy after each epoch
      t_hist   — cumulative wall-clock time (s) after each epoch
    """
    n_tr = X_tr.shape[0]
    W    = np.zeros((C, d))

    tr_acc = np.zeros(epochs)
    va_acc = np.zeros(epochs)
    t_hist = np.zeros(epochs)
    t0     = time.time()

    for epoch in range(epochs):
        # ── Shuffle training data ──────────────────────────────────────────────
        perm  = np.random.permutation(n_tr)
        X_sh  = X_tr[perm]
        y_sh  = y_tr[perm]

        # ── Mini-batch gradient updates ────────────────────────────────────────
        for start in range(0, n_tr, batch_size):
            X_b = X_sh[start : start + batch_size]   # (B, d)
            y_b = y_sh[start : start + batch_size]   # (B,)
            B   = len(y_b)

            # Logits & softmax
            logits = X_b @ W.T                        # (B, C)
            P      = softmax_batch(logits)             # (B, C)

            # One-hot targets
            E      = np.zeros_like(P)
            E[np.arange(B), y_b] = 1.0

            # Gradient: (C, d)
            dW = (P - E).T @ X_b / B

            W -= lr * dW

        # ── Record end-of-epoch stats ──────────────────────────────────────────
        P_tr = softmax_batch(X_tr @ W.T)
        P_va = softmax_batch(X_va @ W.T)
        tr_acc[epoch] = accuracy(y_tr, np.argmax(P_tr, axis=1))
        va_acc[epoch] = accuracy(y_va, np.argmax(P_va, axis=1))
        t_hist[epoch] = time.time() - t0

    return W, tr_acc, va_acc, t_hist


# ── Main training loop (all batch sizes × K folds) ───────────────────────────

results = {}   # batch_size → dict with arrays

for bs in BATCH_SIZES:
    print(f"\n{'='*60}")
    print(f" Mini-batch size = {bs}")
    print(f"{'='*60}")

    all_tr_acc  = np.zeros((K_FOLDS, EPOCHS))
    all_va_acc  = np.zeros((K_FOLDS, EPOCHS))
    all_t       = np.zeros((K_FOLDS, EPOCHS))
    final_W     = None   # weight from last fold (for inspection)

    t_start_bs = time.time()

    for k, (tr_idx, va_idx) in enumerate(kfold_split(N, K_FOLDS)):
        X_tr_k = X_shuf[tr_idx]
        y_tr_k = y_shuf[tr_idx]
        X_va_k = X_shuf[va_idx]
        y_va_k = y_shuf[va_idx]

        W_k, tr_k, va_k, t_k = train_fold(
            X_tr_k, y_tr_k, X_va_k, y_va_k, bs, EPOCHS, LR)

        all_tr_acc[k] = tr_k
        all_va_acc[k] = va_k
        all_t[k]      = t_k
        final_W       = W_k

        print(f"  Fold {k+1:2d}/{K_FOLDS}: "
              f"train acc = {tr_k[-1]:.4f}, "
              f"val acc   = {va_k[-1]:.4f}, "
              f"time = {t_k[-1]:.2f}s")

    t_total = time.time() - t_start_bs
    results[bs] = {
        "tr_acc"  : all_tr_acc,   # (K, EPOCHS)
        "va_acc"  : all_va_acc,
        "t"       : all_t,
        "final_W" : final_W,
        "t_total" : t_total,
    }
    print(f"\n  Summary for batch_size={bs}:")
    print(f"    Mean train acc (last epoch): "
          f"{all_tr_acc[:, -1].mean():.4f} ± {all_tr_acc[:, -1].std():.4f}")
    print(f"    Mean val   acc (last epoch): "
          f"{all_va_acc[:, -1].mean():.4f} ± {all_va_acc[:, -1].std():.4f}")
    print(f"    Total wall-clock time:        {t_total:.2f}s")


# ── Summary table ─────────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("Summary Table — Mean ± Std over 10 folds (epoch 500)")
print("=" * 70)
print(f"{'Batch size':>12} | {'Train Acc':>16} | {'Val Acc':>16} | {'Total Time (s)':>15}")
print("-" * 70)
for bs in BATCH_SIZES:
    r = results[bs]
    tr_m = r["tr_acc"][:, -1].mean()
    tr_s = r["tr_acc"][:, -1].std()
    va_m = r["va_acc"][:, -1].mean()
    va_s = r["va_acc"][:, -1].std()
    t_t  = r["t_total"]
    print(f"{bs:>12} | {tr_m:.4f} ± {tr_s:.4f}  | "
          f"{va_m:.4f} ± {va_s:.4f}  | {t_t:>12.2f}s")


# ── Fig. 15: Mean accuracy vs epoch for each batch size ───────────────────────

fig15, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
labels_bs = {1: "Batch=1 (Online SGD)", 36: "Batch=36", 72: "Batch=72"}
epoch_arr  = np.arange(1, EPOCHS + 1)

for ax, bs in zip(axes, BATCH_SIZES):
    r        = results[bs]
    tr_mean  = r["tr_acc"].mean(axis=0)
    tr_std   = r["tr_acc"].std(axis=0)
    va_mean  = r["va_acc"].mean(axis=0)
    va_std   = r["va_acc"].std(axis=0)

    ax.plot(epoch_arr, tr_mean, "r-",    linewidth=1.5, label="Train")
    ax.fill_between(epoch_arr, tr_mean - tr_std, tr_mean + tr_std,
                    color="red", alpha=0.15)
    ax.plot(epoch_arr, va_mean, "b-",    linewidth=1.5, label="Val")
    ax.fill_between(epoch_arr, va_mean - va_std, va_mean + va_std,
                    color="blue", alpha=0.15)

    ax.set_title(labels_bs[bs], fontsize=11)
    ax.set_xlabel("Epoch", fontsize=10)
    if ax is axes[0]:
        ax.set_ylabel("Accuracy", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_ylim(0.3, 1.05)

    final_va = va_mean[-1]
    ax.axhline(final_va, linestyle=":", color="blue", linewidth=1)
    ax.annotate(f"Val={final_va:.3f}", xy=(EPOCHS, final_va),
                xytext=(-80, 8), textcoords="offset points",
                fontsize=8, color="blue")

fig15.suptitle("Fig. 15 — SGD Convergence: Mean Accuracy vs Epoch  "
               "(Wine, K=10, η=0.1, 500 epochs)", fontsize=12)
fig15.tight_layout()
fig15_path = os.path.join(DIR, "fig15_sgd_convergence.png")
fig15.savefig(fig15_path, dpi=150)
plt.close(fig15)
print(f"\nFig. 15 saved → {fig15_path}")


# ── Fig. 16: Mean val accuracy vs wall-clock time ─────────────────────────────

fig16, ax16 = plt.subplots(figsize=(9, 5))
colors_bs = {1: "darkorange", 36: "steelblue", 72: "seagreen"}

for bs in BATCH_SIZES:
    r       = results[bs]
    t_mean  = r["t"].mean(axis=0)
    va_mean = r["va_acc"].mean(axis=0)
    va_std  = r["va_acc"].std(axis=0)

    ax16.plot(t_mean, va_mean, color=colors_bs[bs],
              linewidth=1.8, label=f"Batch={bs}")
    ax16.fill_between(t_mean, va_mean - va_std, va_mean + va_std,
                      color=colors_bs[bs], alpha=0.15)

ax16.set_xlabel("Wall-clock time (s)", fontsize=11)
ax16.set_ylabel("Mean Validation Accuracy", fontsize=11)
ax16.set_title("Fig. 16 — Validation Accuracy vs Wall-Clock Time\n"
               "(Wine, K=10 folds, η=0.1)", fontsize=11)
ax16.legend(fontsize=10)
ax16.grid(True, linestyle="--", alpha=0.4)
fig16.tight_layout()
fig16_path = os.path.join(DIR, "fig16_acc_vs_time.png")
fig16.savefig(fig16_path, dpi=150)
plt.close(fig16)
print(f"Fig. 16 saved → {fig16_path}")


# ── Confusion matrix for best batch size (last fold) ─────────────────────────

best_bs = max(BATCH_SIZES, key=lambda b: results[b]["va_acc"][:, -1].mean())
print(f"\nBest mean val acc: batch_size={best_bs}")

# Re-run one full train on 80% / test on 20% for confusion matrix display
n_tr_cm  = int(0.8 * N)
X_tr_cm  = X_shuf[:n_tr_cm]
y_tr_cm  = y_shuf[:n_tr_cm]
X_te_cm  = X_shuf[n_tr_cm:]
y_te_cm  = y_shuf[n_tr_cm:]
_, _, va_cm, _ = train_fold(X_tr_cm, y_tr_cm, X_te_cm, y_te_cm,
                             best_bs, EPOCHS, LR)
W_cm  = results[best_bs]["final_W"]
P_cm  = softmax_batch(X_te_cm @ W_cm.T)
y_hat = np.argmax(P_cm, axis=1)
CM    = confusion_matrix(y_te_cm, y_hat, n_classes=3)

print(f"\nConfusion matrix (test set, batch_size={best_bs}, last fold W):")
print("  Rows = True class,  Cols = Predicted class")
class_names = ["Class 0", "Class 1", "Class 2"]
df_cm = pd.DataFrame(CM, index=class_names, columns=class_names)
print(df_cm.to_string())
print(f"  Test accuracy: {accuracy(y_te_cm, y_hat):.4f}")
