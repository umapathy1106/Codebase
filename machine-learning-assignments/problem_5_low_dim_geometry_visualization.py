import numpy as np
import matplotlib.pyplot as plt
from problem_0_data_preparation_mnist import X_tilde, y_selected, mu

# ---------------------------------------------------------
# Problem 5: Low-Dimensional Geometry and Visualization
# ---------------------------------------------------------

classes = [0, 1, 9]
colors = {0: 'blue', 1: 'red', 9: 'green'}
labels = {0: 'Digit 0', 1: 'Digit 1', 9: 'Digit 9'}

# Compute SVD (economy-size)
print("Computing SVD of X_tilde...")
U, S, Vt = np.linalg.svd(X_tilde, full_matrices=False)

# ==========================================================
# Part 1: Project onto first 2 left singular vectors
# ==========================================================
Q2 = U[:, :2]  # (784, 2)
P2 = Q2.T @ X_tilde  # (2, N) — 2D projection of all data

print(f"Q2 shape: {Q2.shape}")
print(f"P2 shape: {P2.shape}")

# ==========================================================
# Part 2: Compute class centroids in projected space
# ==========================================================
centroids_2d = {}
for c in classes:
    mask = (y_selected == c)
    P2_c = P2[:, mask]
    centroids_2d[c] = np.mean(P2_c, axis=1)
    print(f"Centroid for digit {c} in 2D: ({centroids_2d[c][0]:.2f}, {centroids_2d[c][1]:.2f})")

# ==========================================================
# Fig. 5: Scatter plot of P2, color-coded by digit
# ==========================================================
plt.figure(figsize=(10, 8))

for c in classes:
    mask = (y_selected == c)
    plt.scatter(P2[0, mask], P2[1, mask],
                c=colors[c], label=labels[c],
                alpha=0.15, s=5, edgecolors='none')

# Plot centroids with larger markers
for c in classes:
    plt.scatter(centroids_2d[c][0], centroids_2d[c][1],
                c=colors[c], marker='X', s=300,
                edgecolors='black', linewidths=2,
                label=f'Centroid {c}', zorder=5)

plt.xlabel("1st Principal Component (u₁)")
plt.ylabel("2nd Principal Component (u₂)")
plt.title("Fig. 5: 2D Projection of Centered Data onto First Two Left Singular Vectors")
plt.legend(loc='best')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ==========================================================
# Part 3: Reconstruct representative samples for K = 5, 20, 100
# ==========================================================
K_values = [5, 20, 100]

# Pick one representative sample per class (closest to centroid in full space)
representative_idx = {}
for c in classes:
    mask = (y_selected == c)
    indices = np.where(mask)[0]
    X_c = X_tilde[:, mask]
    centroid_c = np.mean(X_c, axis=1, keepdims=True)
    dists = np.linalg.norm(X_c - centroid_c, axis=0)
    best_local = np.argmin(dists)
    representative_idx[c] = indices[best_local]
    print(f"Representative sample for digit {c}: index {representative_idx[c]}")

# ==========================================================
# Fig. 6: Original vs Reconstructed images
# ==========================================================
fig, axes = plt.subplots(len(classes), len(K_values) + 1, figsize=(14, 10))

for row, c in enumerate(classes):
    idx = representative_idx[c]
    x_tilde_sample = X_tilde[:, idx]  # centered image (784,)

    # Original image (add back global mean for visualization)
    original = x_tilde_sample + mu.flatten()

    # Column 0: Original
    axes[row, 0].imshow(original.reshape(28, 28), cmap='gray', vmin=0, vmax=255)
    axes[row, 0].set_title(f"Digit {c}\nOriginal")
    axes[row, 0].axis('off')

    for col, K in enumerate(K_values):
        Q_K = U[:, :K]
        # Reconstruct: project and lift back, then add mean
        reconstructed_centered = Q_K @ (Q_K.T @ x_tilde_sample)
        reconstructed = reconstructed_centered + mu.flatten()

        # Reconstruction error
        recon_error = np.linalg.norm(x_tilde_sample - reconstructed_centered)

        axes[row, col + 1].imshow(reconstructed.reshape(28, 28), cmap='gray', vmin=0, vmax=255)
        axes[row, col + 1].set_title(f"K = {K}\nerror = {recon_error:.2f}")
        axes[row, col + 1].axis('off')

fig.suptitle("Fig. 6: Original vs Reconstructed Images (K = 5, 20, 100)", fontsize=14, y=1.02)
plt.tight_layout()
plt.show()

# ==========================================================
# Discussion
# ==========================================================
print("\n" + "=" * 65)
print("DISCUSSION: Geometric Structure & Reconstruction Behavior")
print("=" * 65)

print(f"\n1. GEOMETRIC STRUCTURE (Fig. 5):")
print(f"   The 2D scatter plot reveals that the three digit classes form")
print(f"   distinct clusters in the space of the first two principal")
print(f"   components. Digit 1 is well-separated from digits 0 and 9,")
print(f"   while 0 and 9 have more overlap — consistent with their")
print(f"   visual similarity (both have curved strokes).")

print(f"\n2. RECONSTRUCTION BEHAVIOR (Fig. 6):")
print(f"   - K = 5:  Captures coarse structure (overall shape),")
print(f"     but images are blurry and details are lost.")
print(f"   - K = 20: Recognizable digits with improved clarity.")
print(f"   - K = 100: Near-perfect reconstruction — fine details")
print(f"     like stroke thickness and curvature are preserved.")
print(f"   Error decreases rapidly with K, matching the singular")
print(f"   value decay from Problem 2.")

print(f"\n3. PROPOSED CLASSIFICATION RULE:")
print(f"   For a new (centered) test image x̃, classify it by:")
print(f"   (a) NEAREST CENTROID IN PROJECTION: Project x̃ to 2D via")
print(f"       p = Q₂ᵀx̃, then assign to the class whose 2D centroid")
print(f"       is closest: argmin_c ||p - m_c||₂.")
print(f"   (b) MINIMUM RECONSTRUCTION ERROR: For each class c, build a")
print(f"       class-specific basis Q_K^(c) from SVD of X̃_c. Assign x̃")
print(f"       to the class whose basis best reconstructs it:")
print(f"       argmin_c ||x̃ - Q_K^(c) (Q_K^(c))ᵀ x̃||₂.")
print(f"   Method (b) is more powerful as it uses class-specific")
print(f"   subspaces rather than a shared global projection.")
