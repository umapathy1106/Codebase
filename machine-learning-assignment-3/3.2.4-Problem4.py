"""
Part II — Classification, SGD, Cross-Validation
3.2.4 Problem 4

Produces three figures from the mini-batch SGD + K-fold CV experiment:
  Fig. 15  — Train & test accuracy vs epoch (solid=train, dashed=test)
  Fig. 16  — Per-epoch wall-clock time vs epoch index
  Fig. 17  — Final average confusion matrices (one per batch size)
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

# ── Utility functions ─────────────────────────────────────────────────────────

def softmax_batch(logits):
    shifted = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(shifted)
    return e / e.sum(axis=1, keepdims=True)


def kfold_split(n, K):
    indices    = np.arange(n)
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


def confusion_matrix_manual(y_true, y_pred, n_classes):
    CM = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        CM[t, p] += 1
    return CM


# ── Load & prepare data ───────────────────────────────────────────────────────

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

df     = pd.read_csv(url, header=None, names=col_names)
X_full = df[feature_names].values.astype(float)
y_full = df["class"].values.astype(int) - 1

# First 40 per class
sel_idx = np.concatenate([np.where(y_full == c)[0][:40] for c in range(3)])
X_filt  = X_full[sel_idx]
y_filt  = y_full[sel_idx]

mu     = X_filt.mean(axis=0)
sigma  = X_filt.std(axis=0)
X_norm = (X_filt - mu) / sigma
X_aug  = np.hstack([X_norm, np.ones((X_norm.shape[0], 1))])   # (120, 14)

C, d = 3, X_aug.shape[1]
N    = X_aug.shape[0]       # 120

# Pre-shuffle (same seed → same split as Problem 3)
preshuf = np.random.permutation(N)
X_shuf  = X_aug[preshuf]
y_shuf  = y_filt[preshuf]

print(f"Dataset ready: X_aug={X_aug.shape}, N={N}")

# ── Hyperparameters ───────────────────────────────────────────────────────────

LR          = 0.1
EPOCHS      = 500
BATCH_SIZES = [1, 36, 72]
K_FOLDS     = 10
CLASS_NAMES = ["Class 0", "Class 1", "Class 2"]

# ── Training function — records per-epoch time ────────────────────────────────

def train_fold(X_tr, y_tr, X_va, y_va, batch_size, epochs, lr):
    """
    Returns
    -------
    tr_acc    : (epochs,)  training accuracy per epoch
    va_acc    : (epochs,)  validation accuracy per epoch
    epoch_t   : (epochs,)  wall-clock time FOR each epoch (seconds)
    W         : (C, d)     final weight matrix
    y_va_pred : (n_va,)    predictions on validation set (last epoch)
    """
    n_tr = X_tr.shape[0]
    W    = np.zeros((C, d))

    tr_acc  = np.zeros(epochs)
    va_acc  = np.zeros(epochs)
    epoch_t = np.zeros(epochs)

    for epoch in range(epochs):
        t_ep_start = time.time()

        perm  = np.random.permutation(n_tr)
        X_sh  = X_tr[perm]
        y_sh  = y_tr[perm]

        for start in range(0, n_tr, batch_size):
            X_b = X_sh[start : start + batch_size]
            y_b = y_sh[start : start + batch_size]
            B   = len(y_b)

            P          = softmax_batch(X_b @ W.T)
            E          = np.zeros_like(P)
            E[np.arange(B), y_b] = 1.0
            W         -= lr * ((P - E).T @ X_b / B)

        epoch_t[epoch] = time.time() - t_ep_start

        P_tr = softmax_batch(X_tr @ W.T)
        P_va = softmax_batch(X_va @ W.T)
        tr_acc[epoch] = accuracy(y_tr, np.argmax(P_tr, axis=1))
        va_acc[epoch] = accuracy(y_va, np.argmax(P_va, axis=1))

    y_va_pred = np.argmax(softmax_batch(X_va @ W.T), axis=1)
    return tr_acc, va_acc, epoch_t, W, y_va_pred


# ── Run all experiments ───────────────────────────────────────────────────────

results = {}

for bs in BATCH_SIZES:
    print(f"\nBatch size = {bs}")
    all_tr  = np.zeros((K_FOLDS, EPOCHS))
    all_va  = np.zeros((K_FOLDS, EPOCHS))
    all_et  = np.zeros((K_FOLDS, EPOCHS))
    CM_sum  = np.zeros((C, C), dtype=int)

    for k, (tr_idx, va_idx) in enumerate(kfold_split(N, K_FOLDS)):
        X_tr_k = X_shuf[tr_idx];  y_tr_k = y_shuf[tr_idx]
        X_va_k = X_shuf[va_idx];  y_va_k = y_shuf[va_idx]

        tr_k, va_k, et_k, _, y_pred_k = train_fold(
            X_tr_k, y_tr_k, X_va_k, y_va_k, bs, EPOCHS, LR)

        all_tr[k]  = tr_k
        all_va[k]  = va_k
        all_et[k]  = et_k
        CM_sum    += confusion_matrix_manual(y_va_k, y_pred_k, C)

        print(f"  Fold {k+1:2d}: val_acc={va_k[-1]:.4f}, "
              f"mean epoch time={et_k.mean()*1000:.2f}ms")

    results[bs] = {
        "tr"  : all_tr,   # (K, EPOCHS)
        "va"  : all_va,
        "et"  : all_et,   # per-epoch time
        "CM"  : CM_sum,   # summed confusion matrix over all folds
    }
    print(f"  Mean val acc = {all_va[:,-1].mean():.4f}  |  "
          f"CM diagonal sum = {CM_sum.diagonal().sum()} / {N}")


# ─────────────────────────────────────────────────────────────────────────────
# Fig. 15 — Train & Test accuracy vs epoch (solid=train, dashed=test)
# ─────────────────────────────────────────────────────────────────────────────

COLORS  = {1: "darkorange", 36: "steelblue", 72: "seagreen"}
epoch_x = np.arange(1, EPOCHS + 1)

fig15, ax15 = plt.subplots(figsize=(10, 5))

for bs in BATCH_SIZES:
    r       = results[bs]
    tr_mean = r["tr"].mean(axis=0)
    va_mean = r["va"].mean(axis=0)
    c       = COLORS[bs]
    ax15.plot(epoch_x, tr_mean, color=c, linestyle="-",
              linewidth=1.8, label=f"Train  (B={bs})")
    ax15.plot(epoch_x, va_mean, color=c, linestyle="--",
              linewidth=1.8, label=f"Test   (B={bs})")

ax15.set_xlabel("Epoch", fontsize=12)
ax15.set_ylabel("Accuracy", fontsize=12)
ax15.set_title("Fig. 15 — Train & Test Accuracy vs Epoch\n"
               "(Wine, K=10, η=0.1, solid=Train, dashed=Test)", fontsize=11)
ax15.legend(fontsize=9, ncol=2)
ax15.grid(True, linestyle="--", alpha=0.4)
ax15.set_ylim(0.5, 1.05)
fig15.tight_layout()
p15 = os.path.join(DIR, "fig15_acc_vs_epoch.png")
fig15.savefig(p15, dpi=150)
plt.close(fig15)
print(f"\nFig. 15 saved → {p15}")


# ─────────────────────────────────────────────────────────────────────────────
# Fig. 16 — Per-epoch wall-clock time vs epoch index
# ─────────────────────────────────────────────────────────────────────────────

fig16, ax16 = plt.subplots(figsize=(10, 5))

for bs in BATCH_SIZES:
    r      = results[bs]
    t_mean = r["et"].mean(axis=0)          # mean over folds
    t_std  = r["et"].std(axis=0)
    c      = COLORS[bs]
    ax16.plot(epoch_x, t_mean * 1000, color=c, linewidth=1.5,
              label=f"Batch={bs}")
    ax16.fill_between(epoch_x,
                      (t_mean - t_std) * 1000,
                      (t_mean + t_std) * 1000,
                      color=c, alpha=0.15)

ax16.set_xlabel("Epoch", fontsize=12)
ax16.set_ylabel("Time per epoch (ms)", fontsize=12)
ax16.set_title("Fig. 16 — Per-Epoch Wall-Clock Time vs Epoch Index\n"
               "(Wine, K=10, η=0.1, shaded=±1σ across folds)", fontsize=11)
ax16.legend(fontsize=10)
ax16.grid(True, linestyle="--", alpha=0.4)
fig16.tight_layout()
p16 = os.path.join(DIR, "fig16_time_vs_epoch.png")
fig16.savefig(p16, dpi=150)
plt.close(fig16)
print(f"Fig. 16 saved → {p16}")


# ─────────────────────────────────────────────────────────────────────────────
# Fig. 17 — Final confusion matrices (one per batch size)
# ─────────────────────────────────────────────────────────────────────────────

fig17, axes17 = plt.subplots(1, 3, figsize=(13, 4))

for ax, bs in zip(axes17, BATCH_SIZES):
    CM = results[bs]["CM"]   # summed over 10 folds (= all 120 validation preds)

    # Display as heatmap
    im = ax.imshow(CM, cmap="Blues", aspect="auto")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(C))
    ax.set_yticks(range(C))
    ax.set_xticklabels(CLASS_NAMES, fontsize=9)
    ax.set_yticklabels(CLASS_NAMES, fontsize=9)
    ax.set_xlabel("Predicted class", fontsize=10)
    ax.set_ylabel("True class", fontsize=10)
    ax.set_title(f"Batch size = {bs}\n"
                 f"Accuracy = {CM.diagonal().sum()/CM.sum():.4f}", fontsize=10)

    # Annotate each cell with the count
    thresh = CM.max() / 2.0
    for i in range(C):
        for j in range(C):
            ax.text(j, i, str(CM[i, j]),
                    ha="center", va="center", fontsize=13,
                    color="white" if CM[i, j] > thresh else "black",
                    fontweight="bold")

fig17.suptitle("Fig. 17 — Final Confusion Matrices  "
               "(summed over K=10 folds, all 120 val predictions)\n"
               "Rows = True class,  Columns = Predicted class",
               fontsize=11)
fig17.tight_layout()
p17 = os.path.join(DIR, "fig17_confusion_matrices.png")
fig17.savefig(p17, dpi=150)
plt.close(fig17)
print(f"Fig. 17 saved → {p17}")


# ── Print confusion matrix tables ─────────────────────────────────────────────

print("\n" + "=" * 60)
print("Confusion Matrices (summed over 10 folds)")
print("  Rows = True class;  Columns = Predicted class")
print("=" * 60)
for bs in BATCH_SIZES:
    CM  = results[bs]["CM"]
    acc = CM.diagonal().sum() / CM.sum()
    print(f"\nBatch size = {bs}  (overall accuracy = {acc:.4f})")
    df_cm = pd.DataFrame(CM, index=CLASS_NAMES, columns=CLASS_NAMES)
    print(df_cm.to_string())
