import numpy as np
import matplotlib.pyplot as plt
from problem_0_data_preparation_mnist import X_tilde

# ---------------------------------------------------------
# Problem 2: Singular Value Decomposition
# ---------------------------------------------------------

# 1. Compute the full SVD of X_tilde (784 x N)
#    X_tilde = U @ Sigma @ V^T
#    U: (784 x 784), orthonormal columns
#    S: singular values (min(784, N),)
#    Vt: (N x N), orthonormal rows (V^T)
print("Computing SVD of X_tilde... (this may take a moment)")
U, S, Vt = np.linalg.svd(X_tilde, full_matrices=True)

print(f"U shape:  {U.shape}")    # (784, 784)
print(f"S shape:  {S.shape}")    # (min(784, N),)
print(f"Vt shape: {Vt.shape}")   # (N, N)

# 2. Determine the rank r = number of nonzero singular values
tol = 1e-6
r = np.sum(S > tol)
print(f"\nRank of X_tilde (r): {r}")
print(f"Largest singular value  σ_1 = {S[0]:.4f}")
print(f"Smallest nonzero σ_r   σ_{r} = {S[r-1]:.4f}")

# ---------------------------------------------------------
# Fig. 1: Singular values σ_i vs index i
# ---------------------------------------------------------
plt.figure(figsize=(10, 5))
plt.plot(range(1, r + 1), S[:r], 'b-', linewidth=1.5)
plt.xlabel("Index i")
plt.ylabel("Singular Value σ_i")
plt.title("Fig. 1: Singular Value Spectrum of X̃")
plt.grid(True, alpha=0.3)
plt.yscale('log')
plt.xlim([1, r])
plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# Fig. 2: Cumulative energy E(K) vs K
# ---------------------------------------------------------
# E(K) = sum(σ_i^2, i=1..K) / sum(σ_i^2, i=1..r)
sigma_sq = S[:r] ** 2
total_energy = np.sum(sigma_sq)
cumulative_energy = np.cumsum(sigma_sq) / total_energy

plt.figure(figsize=(10, 5))
plt.plot(range(1, r + 1), cumulative_energy, 'r-', linewidth=1.5)
plt.xlabel("K (number of singular values)")
plt.ylabel("Cumulative Energy E(K)")
plt.title("Fig. 2: Cumulative Energy (Variance Explained) Ratio")
plt.grid(True, alpha=0.3)
plt.axhline(y=0.90, color='gray', linestyle='--', alpha=0.7, label='90% energy')
plt.axhline(y=0.95, color='gray', linestyle=':', alpha=0.7, label='95% energy')
plt.axhline(y=0.99, color='gray', linestyle='-.', alpha=0.7, label='99% energy')
plt.legend()
plt.xlim([1, r])
plt.ylim([0, 1.02])
plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# Discussion
# ---------------------------------------------------------
# Find K for key energy thresholds
K_90 = np.searchsorted(cumulative_energy, 0.90) + 1
K_95 = np.searchsorted(cumulative_energy, 0.95) + 1
K_99 = np.searchsorted(cumulative_energy, 0.99) + 1

print("\n" + "=" * 60)
print("DISCUSSION: Singular Value Decay & Energy Curve")
print("=" * 60)
print(f"\nEnergy thresholds:")
print(f"  90% variance explained at K = {K_90}  (out of r = {r})")
print(f"  95% variance explained at K = {K_95}  (out of r = {r})")
print(f"  99% variance explained at K = {K_99}  (out of r = {r})")
print(f"\n1. VARIANCE: Each σ_i^2 represents the variance captured by")
print(f"   the i-th principal component. The first few singular values")
print(f"   are very large, meaning most variance is concentrated in a")
print(f"   small number of directions.")
print(f"\n2. REDUNDANCY: The rapid decay of singular values shows that")
print(f"   the 784-dimensional pixel space is highly redundant for")
print(f"   representing digits 0, 1, 9. Most pixel values are strongly")
print(f"   correlated (e.g., neighboring pixels tend to be similar).")
print(f"\n3. INTRINSIC DIMENSIONALITY: Only K = {K_95} components are")
print(f"   needed to capture 95% of the total variance, despite the")
print(f"   data living in R^784. This suggests the intrinsic")
print(f"   dimensionality of the digit data is roughly {K_95},")
print(f"   far smaller than 784.")
