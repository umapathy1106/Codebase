import numpy as np
import matplotlib.pyplot as plt
from problem_0_data_preparation_mnist import X_tilde, y_selected

# ---------------------------------------------------------
# Problem 1: Norms and Centroids
# ---------------------------------------------------------

# We will compute the centroids and norms for digits 0, 1, and 9
# The centroids are computed as the mean of the centered data for each class,
# and the norms are computed for these centroids.

classes = [0, 1, 9]
centroids = {} # Dictionary to store centroids for each class
norms = {} # Dictionary to store norms for each class

print(f"{'Digit':<6} | {'L1 Norm':<12} | {'L2 Norm':<12} | {'L3 Norm':<12}")
print("-" * 50)

plt.figure(figsize=(12, 4))

# Loop through each class to compute centroids and norms
# We will also visualize the centroids as images to see how they differ from the global mean.
for i, c in enumerate(classes):
    # 1. Extract submatrix X_tilde_c
    # We use the mask (y_selected == c) to select columns
    class_mask = (y_selected == c)
    X_tilde_c = X_tilde[:, class_mask]
    
    # 2. Compute Centroid m_c
    # Average across columns (axis=1)
    m_c = np.mean(X_tilde_c, axis=1)
    centroids[c] = m_c
    
    # 3. Compute Norms
    # L1 norm is the sum of absolute values, L2 norm is the square root of the sum of squares, and L3 norm is the cube root of the sum of cubes of absolute values.
    l1 = np.sum(np.abs(m_c)) # L1 norm is the sum of absolute values
    l2 = np.linalg.norm(m_c, 2) # L2 norm is the square root of the sum of squares
    l3 = np.power(np.sum(np.power(np.abs(m_c), 3)), 1/3)
    
    # Store the norms in the dictionary
    norms[c] = (l1, l2, l3)
    
    print(f"{c:<6} | {l1:<12.2f} | {l2:<12.2f} | {l3:<12.2f}")

    # Visualize the Centroid (Difference Image)
    plt.subplot(1, 3, i+1)
    # Reshape 784 -> 28x28 for visualization
    plt.imshow(m_c.reshape(28, 28), cmap='seismic', vmin=-np.max(np.abs(m_c)), vmax=np.max(np.abs(m_c)))
    plt.title(f"Centroid Digit {c}\n(Deviation from Global Mean)")
    plt.axis('off')
    plt.colorbar(fraction=0.046, pad=0.04)

plt.tight_layout() # Adjust layout to prevent overlap of subplots
plt.show() # Display the centroids as images to visualize the deviation from the global mean for each digit class.