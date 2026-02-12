import numpy as np
import matplotlib.pyplot as plt
from problem_0_data_preparation_mnist import X, X_tilde, mu

# ---------------------------------------------------------
# Problem 6: Low-Rank Denoising
# ---------------------------------------------------------

d, N = X.shape  # d=784, N=number of images
print(f"Data dimensions: d = {d}, N = {N}")

# ==========================================================
# Step 1: Construct noisy data matrix X_n
# ==========================================================
# Independently corrupt each pixel with probability 0.05
# Replace corrupted pixel with 0 or 255 (equal probability)
np.random.seed(42)  # for reproducibility

corruption_prob = 0.05
corruption_mask = np.random.rand(d, N) < corruption_prob  # True where corrupted
salt_or_pepper = np.random.rand(d, N) < 0.5  # True -> 255, False -> 0

X_n = X.copy()
X_n[corruption_mask & salt_or_pepper] = 255.0
X_n[corruption_mask & ~salt_or_pepper] = 0.0

num_corrupted = np.sum(corruption_mask)
print(f"Number of corrupted pixels: {num_corrupted:,} out of {d * N:,} "
      f"({num_corrupted / (d * N) * 100:.2f}%)")

# Compute the baseline error: e(X, X_n)
error_X_Xn = np.linalg.norm(X - X_n, 'fro')
print(f"Baseline noise error e(X, X_n) = {error_X_Xn:.4f}")

# ==========================================================
# Step 2: Compute SVD of X_tilde (from clean centered data)
# ==========================================================
# The SVD basis Q_K comes from the clean centered data
print("Computing SVD of X_tilde (clean centered data)...")
U, S, Vt = np.linalg.svd(X_tilde, full_matrices=False)

# ==========================================================
# Step 3: For each K, compute Y_K = Q_K @ Q_K^T @ X_n
#         and errors e(X, Y_K), e(X_n, Y_K), e(X, X_n)
# ==========================================================
K_set = list(range(1, 785, 50))
if K_set[-1] != 784:
    K_set.append(784)

errors_X_YK = []     # e(X, Y_K) — how close reconstruction is to clean data
errors_Xn_YK = []    # e(X_n, Y_K) — how close reconstruction is to noisy data
errors_X_Xn_list = []  # e(X, X_n) — constant baseline

print(f"\n{'K':<6} | {'e(X, Y_K)':<16} | {'e(X_n, Y_K)':<16} | {'e(X, X_n)':<16}")
print("-" * 60)

for K in K_set:
    Q_K = U[:, :K]

    # Y_K = Q_K @ Q_K^T @ X_n  (project noisy data onto clean basis)
    Y_K = Q_K @ (Q_K.T @ X_n)

    e_X_YK = np.linalg.norm(X - Y_K, 'fro')
    e_Xn_YK = np.linalg.norm(X_n - Y_K, 'fro')

    errors_X_YK.append(e_X_YK)
    errors_Xn_YK.append(e_Xn_YK)
    errors_X_Xn_list.append(error_X_Xn)

    print(f"{K:<6} | {e_X_YK:<16.4f} | {e_Xn_YK:<16.4f} | {error_X_Xn:<16.4f}")

# ==========================================================
# Fig. 7: All errors vs K
# ==========================================================
plt.figure(figsize=(10, 6))
plt.plot(K_set, errors_X_YK, 'b-o', linewidth=1.5, markersize=5, label='e(X, Y_K) — reconstruction vs clean')
plt.plot(K_set, errors_Xn_YK, 'g-s', linewidth=1.5, markersize=5, label='e(X_n, Y_K) — reconstruction vs noisy')
plt.axhline(y=error_X_Xn, color='r', linestyle='--', linewidth=2, label=f'e(X, X_n) = {error_X_Xn:.2f} (noise level)')
plt.xlabel("K (rank of approximation)")
plt.ylabel("Frobenius Norm Error")
plt.title("Fig. 7: Low-Rank Denoising — Error Curves vs K")
plt.legend()
plt.grid(True, alpha=0.3)
plt.xlim([0, 790])
plt.tight_layout()
plt.show()

# ==========================================================
# Bonus: Visualize denoising on a sample image
# ==========================================================
# Pick a sample from digit 0
sample_idx = np.where(np.array([0]) == 0)[0][0]  # first image
sample_idx = 0

fig, axes = plt.subplots(1, 5, figsize=(16, 3.5))

# Original
axes[0].imshow(X[:, sample_idx].reshape(28, 28), cmap='gray', vmin=0, vmax=255)
axes[0].set_title("Original")
axes[0].axis('off')

# Noisy
axes[1].imshow(X_n[:, sample_idx].reshape(28, 28), cmap='gray', vmin=0, vmax=255)
axes[1].set_title("Noisy (5% corruption)")
axes[1].axis('off')

# Denoised at K=5, 20, 100
for i, K in enumerate([5, 20, 100]):
    Q_K = U[:, :K]
    y_k = Q_K @ (Q_K.T @ X_n[:, sample_idx])
    axes[i + 2].imshow(np.clip(y_k.reshape(28, 28), 0, 255), cmap='gray', vmin=0, vmax=255)
    axes[i + 2].set_title(f"Denoised K={K}")
    axes[i + 2].axis('off')

plt.suptitle("Denoising Example: Original → Noisy → Low-Rank Reconstructions")
plt.tight_layout()
plt.show()

# ==========================================================
# Discussion
# ==========================================================
# Find optimal K (minimum of e(X, Y_K))
optimal_idx = np.argmin(errors_X_YK)
optimal_K = K_set[optimal_idx]
optimal_error = errors_X_YK[optimal_idx]

print("\n" + "=" * 65)
print("DISCUSSION: Low-Rank Denoising Behavior")
print("=" * 65)

print(f"\nOptimal K = {optimal_K} with e(X, Y_K) = {optimal_error:.4f}")
print(f"Noise level e(X, X_n) = {error_X_Xn:.4f}")
print(f"Denoising improvement: {((error_X_Xn - optimal_error) / error_X_Xn * 100):.1f}% error reduction")

print(f"\n1. e(X, Y_K) — RECONSTRUCTION vs CLEAN DATA (blue curve):")
print(f"   This curve has a U-shape (or decreases then plateaus).")
print(f"   - For small K: error is high because the low-rank basis")
print(f"     cannot represent the clean signal well (underfitting).")
print(f"   - For moderate K: error reaches its minimum — the basis")
print(f"     captures the signal but rejects the noise.")
print(f"   - For large K: error increases again because the projection")
print(f"     starts to faithfully reproduce the noise (overfitting).")

print(f"\n2. e(X_n, Y_K) — RECONSTRUCTION vs NOISY DATA (green curve):")
print(f"   Monotonically decreases as K increases. At K=784, Y_K = X_n")
print(f"   exactly (zero error). This curve measures fitting, not")
print(f"   denoising — low error here does NOT mean good denoising.")

print(f"\n3. e(X, X_n) — NOISE LEVEL (red dashed line):")
print(f"   Constant baseline. When e(X, Y_K) < e(X, X_n), the low-rank")
print(f"   approximation is closer to the clean data than the noisy")
print(f"   data is — meaning denoising is effective.")

print(f"\n4. WHY IT WORKS: Noise is high-rank (spread uniformly across")
print(f"   all singular value directions), while the signal is low-rank")
print(f"   (concentrated in the top singular values). A low-rank")
print(f"   projection keeps the signal and discards the noise.")
