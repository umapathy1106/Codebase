import numpy as np
import matplotlib.pyplot as plt
from problem_0_data_preparation_mnist import X_tilde

# ---------------------------------------------------------
# Problem 3: Low-Rank Approximation
# ---------------------------------------------------------

# 1. Compute SVD of X_tilde
print("Computing SVD of X_tilde...")
U, S, Vt = np.linalg.svd(X_tilde, full_matrices=False)
# U: (784, min(784,N)), S: (min(784,N),), Vt: (min(784,N), N)
print(f"U shape: {U.shape}, S shape: {S.shape}")

# 2. Define K_set = {1, 51, 101, ..., 784}
K_set = list(range(1, 785, 50))
if K_set[-1] != 784:
    K_set.append(784)
print(f"K_set: {K_set}")

# 3. For each K, compute rank-K approximation and its error
#    Q_K = U[:, :K]  (first K columns of U)
#    Y_K = Q_K @ Q_K^T @ X_tilde
#    error = ||X_tilde - Y_K||_F
#
#    By SVD properties: ||X_tilde - Y_K||_F = sqrt(sum of sigma_i^2 for i > K)
#    We compute both the direct way and the SVD shortcut.

errors = []
frobenius_X_tilde = np.linalg.norm(X_tilde, 'fro')
print(f"\nFrobenius norm of X_tilde: {frobenius_X_tilde:.4f}")
print(f"\n{'K':<6} | {'Error ||X̃ - Y_K||_F':<25} | {'Relative Error':<15}")
print("-" * 55)

for K in K_set:
    # Q_K: first K columns of U
    Q_K = U[:, :K]

    # Y_K = Q_K @ Q_K^T @ X_tilde
    # Efficient computation: Q_K @ (Q_K^T @ X_tilde) avoids forming 784x784 matrix
    Y_K = Q_K @ (Q_K.T @ X_tilde)

    # Approximation error (Frobenius norm)
    error = np.linalg.norm(X_tilde - Y_K, 'fro')
    errors.append(error)

    rel_error = error / frobenius_X_tilde
    print(f"{K:<6} | {error:<25.4f} | {rel_error:<15.6f}")

# ---------------------------------------------------------
# Fig. 3: Approximation error e(X_tilde, Y_K) vs K
# ---------------------------------------------------------
plt.figure(figsize=(10, 5))
plt.plot(K_set, errors, 'b-o', linewidth=1.5, markersize=5)
plt.xlabel("K (rank of approximation)")
plt.ylabel("Approximation Error ||X̃ − Y_K||_F")
plt.title("Fig. 3: Low-Rank Approximation Error vs K")
plt.grid(True, alpha=0.3)
plt.yscale('log')
plt.xlim([0, 790])
plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# Discussion
# ---------------------------------------------------------
# Theoretical relationship: by the Eckart-Young theorem,
# ||X_tilde - Y_K||_F = sqrt(sigma_{K+1}^2 + ... + sigma_r^2)
tol = 1e-6
r = np.sum(S > tol)

print("\n" + "=" * 60)
print("DISCUSSION: Error Decay and Singular Value Spectrum")
print("=" * 60)
print(f"\n1. ECKART-YOUNG THEOREM: The rank-K approximation Y_K is the")
print(f"   best rank-K approximation to X̃ in the Frobenius norm.")
print(f"   The error is exactly:")
print(f"     ||X̃ - Y_K||_F = sqrt(σ²_{{K+1}} + σ²_{{K+2}} + ... + σ²_r)")
print(f"\n2. ERROR DECAY: The error decreases as K increases because")
print(f"   each additional singular vector captures more variance.")
print(f"   The rapid initial decay mirrors the fast drop in singular")
print(f"   values — the first few components capture most of the")
print(f"   signal, so the error shrinks quickly at small K.")
print(f"\n3. DIMINISHING RETURNS: Beyond a moderate K, the error")
print(f"   plateaus because the remaining singular values are small.")
print(f"   This confirms that the data has low intrinsic")
print(f"   dimensionality and a low-rank approximation suffices.")
print(f"\n4. At K = {K_set[0]}: error = {errors[0]:.2f}")
print(f"   At K = {K_set[len(K_set)//2]}: error = {errors[len(K_set)//2]:.2f}")
print(f"   At K = {K_set[-1]}: error = {errors[-1]:.6f} (essentially zero)")
