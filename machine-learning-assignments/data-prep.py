import numpy as np
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

# Load MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Combine train and test sets
X_all = np.concatenate([x_train, x_test], axis=0)
y_all = np.concatenate([y_train, y_test], axis=0)

# Select digits 0, 1, and 9
mask = np.isin(y_all, [0, 1, 9])
X_selected = X_all[mask]
y_selected = y_all[mask]

# Print full data without truncation
np.set_printoptions(threshold=np.inf, suppress=True, linewidth=np.inf)

# Find and display a visual example of digit 9
nine_index = np.where(y_all == 9)[0][0]  # Find first image with label 9
nine_image = X_all[nine_index]

print("=" * 50)
print("VISUAL EXAMPLE: Digit 9")
print("=" * 50)
print(f"Label: {y_all[nine_index]}")
print(f"Shape: {nine_image.shape}")
print(f"Pixel values range: {nine_image.min()} to {nine_image.max()}")
print("\nPixel Data (28x28 grid):")
print(nine_image)

# Display as ASCII art (simple visualization)
print("\nASCII Visualization:")
for row in nine_image:
    for pixel in row:
        # Convert pixel value (0-255) to character using intensity
        if pixel > 200:
            print("@", end="")
        elif pixel > 100:
            print("#", end="")
        elif pixel > 50:
            print(".", end="")
        else:
            print(" ", end="")
    print()

# Save as image file
plt.figure(figsize=(4, 4))
plt.imshow(nine_image, cmap='gray')
plt.title(f"Digit 9 - Label: {y_all[nine_index]}")
plt.colorbar(label='Pixel Intensity')
plt.savefig('digit_9_example.png', dpi=100, bbox_inches='tight')
plt.close()
print("\n✓ Image saved as 'digit_9_example.png'")
print("=" * 50)

# Print X_all
print("Shape:", X_all.shape)
print("Data type:", X_all.dtype)
print(X_all)

# Print y_all
print("\ny_all:")
print("Shape:", y_all.shape)
print(y_all)

# Print X_selected
print("\nX_selected:")
print("Shape:", X_selected.shape)
print(X_selected)

# Print y_selected
print("\ny_selected:")
print("Shape:", y_selected.shape)
print(y_selected)

