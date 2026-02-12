import numpy as np
import matplotlib.pyplot as plt
from problem_0_data_preparation_mnist import X_tilde, X

# ---------------------------------------------------------
# Problem 4: Storage Efficiency
# ---------------------------------------------------------

# Dimensions
d, N = X_tilde.shape  # d = 784, N = number of images
print(f"Data dimensions: d = {d}, N = {N}")

# Compute SVD (economy-size)
print("Computing SVD of X_tilde...")
U, S, Vt = np.linalg.svd(X_tilde, full_matrices=False)

# K_set same as Problem 3
K_set = list(range(1, 785, 50))
if K_set[-1] != 784:
    K_set.append(784)

# ---------------------------------------------------------
# Storage computations
# ---------------------------------------------------------

# Full data matrix X: d x N scalars
storage_X = d * N
print(f"\nStorage for full X: {d} × {N} = {storage_X:,} scalars")

# Low-rank representation: Q_K (d x K) + Z_K (K x N)
# Total = d*K + K*N = K*(d + N)
storage_low_rank = []
errors = []

print(f"\n{'K':<6} | {'Q_K (d×K)':<14} | {'Z_K (K×N)':<14} | {'Total':<14} | {'Ratio vs X':<12} | {'Error':<12}")
print("-" * 80)

for K in K_set:
    Q_K = U[:, :K]
    Z_K = Q_K.T @ X_tilde  # K x N

    storage_QK = d * K
    storage_ZK = K * N
    total_storage = storage_QK + storage_ZK

    # Approximation error
    Y_K = Q_K @ Z_K
    error = np.linalg.norm(X_tilde - Y_K, 'fro')

    storage_low_rank.append(total_storage)
    errors.append(error)

    ratio = total_storage / storage_X
    print(f"{K:<6} | {storage_QK:<14,} | {storage_ZK:<14,} | {total_storage:<14,} | {ratio:<12.4f} | {error:<12.4f}")

# ---------------------------------------------------------
# Fig. 4: Storage cost vs K
# ---------------------------------------------------------
plt.figure(figsize=(10, 5))
plt.plot(K_set, storage_low_rank, 'b-o', linewidth=1.5, markersize=5, label='Low-rank storage K·(d + N)')
plt.axhline(y=storage_X, color='r', linestyle='--', linewidth=2, label=f'Full X storage (d·N = {storage_X:,})')
plt.xlabel("K (rank of approximation)")
plt.ylabel("Number of Scalar Values")
plt.title("Fig. 4: Storage Cost — Low-Rank Representation vs Full Matrix")
plt.legend()
plt.grid(True, alpha=0.3)
plt.xlim([0, 790])
plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# Discussion
# ---------------------------------------------------------
# Find the crossover point where low-rank storage exceeds full storage
crossover_K = None
for i, K in enumerate(K_set):
    if storage_low_rank[i] >= storage_X:
        crossover_K = K
        break

# Find K values for good accuracy/storage tradeoffs
frobenius_X_tilde = np.linalg.norm(X_tilde, 'fro')
rel_errors = [e / frobenius_X_tilde for e in errors]

print("\n" + "=" * 60)
print("DISCUSSION: Approximation Accuracy vs Storage Efficiency")
print("=" * 60)
print(f"\nFull matrix X storage: d × N = {d} × {N} = {storage_X:,} scalars")
print(f"Low-rank storage:      K × (d + N) = K × ({d} + {N}) = K × {d + N}")
if crossover_K:
    print(f"\nCrossover point: low-rank storage exceeds full storage at K = {crossover_K}")
    print(f"  (K_crossover = d·N / (d+N) = {d*N / (d+N):.0f})")
print(f"\nStorage vs Accuracy tradeoffs:")
for i, K in enumerate(K_set):
    if K in [1, 51, 101, 201, 401, 784]:
        savings = (1 - storage_low_rank[i] / storage_X) * 100
        print(f"  K = {K:<4}: storage = {storage_low_rank[i]:>12,} ({savings:>+6.1f}% vs X), "
              f"relative error = {rel_errors[i]:.6f}")
print(f"\nKey insight: For small K, the low-rank factorization (Q_K, Z_K)")
print(f"requires far fewer scalars than storing X directly, while still")
print(f"capturing most of the variance. This is the foundation of")
print(f"dimensionality reduction — we trade a small loss in accuracy")
print(f"for a large reduction in storage.")
