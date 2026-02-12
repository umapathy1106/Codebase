"""
================================================================================
ASSIGNMENT: MNIST Data Preparation for Dimensionality Reduction
================================================================================
OBJECTIVE:
This assignment prepares MNIST handwritten digit data for machine learning analysis.
Specifically, we load the full MNIST dataset, filter for digits 0, 1, and 9,
vectorize the images, compute the global mean, and center the data.

This preprocessing is essential for Principal Component Analysis (PCA) and other
dimensionality reduction techniques that require zero-centered data.
================================================================================
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

# ==============================================================================
# PART 1: LOAD THE MNIST DATASET
# ==============================================================================
# MNIST (Modified National Institute of Science and Technology) dataset:
#   - Contains 70,000 handwritten digit images (0-9)
#   - Split: 60,000 training images + 10,000 test images
#   - Each image: 28×28 pixels (values 0-255 representing grayscale intensity)
#
# STRATEGY: Combine train and test sets to MAXIMIZE N (total number of samples)
# This provides more data for our analysis, improving statistical robustness.
#
# EXPLANATION OF LOADING:
# - mnist.load_data() returns two tuples: (train_data, test_data)
# - Each tuple contains (images, labels)
# - x_train/x_test: Image pixel data (shape: [num_images, 28, 28])
# - y_train/y_test: Image labels (shape: [num_images])

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# CONCATENATION: Merge training and test sets along axis=0 (samples dimension)
# axis=0 means we stack datasets vertically (adding more samples)
# This gives us access to all 70,000 images for processing
x_all = np.concatenate((x_train, x_test), axis=0)  # Shape: (70000, 28, 28)
y_all = np.concatenate((y_train, y_test), axis=0)  # Shape: (70000,)

# ==============================================================================
# PART 2: FILTER FOR DIGITS 0, 1, AND 9
# ==============================================================================
# OBJECTIVE: Reduce the dataset to a specific subset of digits for focused analysis
#
# WHY THESE DIGITS?
# - Digits 0, 1, 9 represent different visual characteristics:
#   * 0: Oval shape with hollow center
#   * 1: Thin vertical line
#   * 9: Rounded top with vertical stem
# - This subset allows for meaningful classification and dimensionality analysis
#
# BOOLEAN MASKING TECHNIQUE:
# np.isin(y_all, [0, 1, 9]) creates a boolean array where:
#   - True: if the label IS in [0, 1, 9]
#   - False: if the label is NOT in [0, 1, 9]
#
# This boolean mask then filters both images and labels simultaneously

mask = np.isin(y_all, [0, 1, 9])  # Boolean array: (70000,)
x_selected = x_all[mask]           # Select only images matching the mask
y_selected = y_all[mask]           # Select only labels matching the mask

# RESULT: We now have only the data for digits 0, 1, and 9
# Expected: ~21,000 images (roughly 1/3 of 70,000, since we kept 3 out of 10 digits)

# ==============================================================================
# PART 3: VECTORIZE AND RESHAPE
# ==============================================================================
# N is the total number of selected images [cite: 24]
# N serves as a parameter throughout our analysis (number of samples)

N = x_selected.shape[0]  # Extract the number of selected images
print(f"Total number of selected images (N): {N}")

# VECTORIZATION PROCESS:
# We need to convert 2D images into 1D feature vectors and organize them
# into a MATRIX FORMAT suitable for linear algebra operations (PCA, SVD, etc.)
#
# STEP-BY-STEP TRANSFORMATION:
# Original shape of x_selected: (N, 28, 28)
#   → Each image is a 28×28 2D grid of pixel values
#
# After .reshape(N, 784):  (N, 784)
#   → Flatten each 28×28 image into a 1D vector of length 784
#   → 784 = 28 × 28 (total number of pixels)
#   → Each row now contains one flattened image
#
# After .T (transpose):  (784, N)
#   → CRUCIAL: Transpose converts rows to columns
#   → NOW: Each COLUMN is one image (observation)
#   → Each ROW represents one pixel feature across all images
#
# After .astype(np.float32):
#   → Convert integer pixel values (0-255) to floating-point numbers
#   → float32 is more efficient and suitable for numerical computations
#   → Required for algorithms like PCA and SVD
#
# MATHEMATICAL NOTATION:
# X ∈ ℝ^(784×N) where:
#   - Rows (784): Feature dimension (pixel values)
#   - Columns (N): Sample dimension (individual images)
#
# This matrix form is standard in machine learning:
#   - Column vectors represent samples/observations
#   - Row vectors represent features/attributes

X = x_selected.reshape(N, 784).T.astype(np.float32)  # Shape: (784, N)

print(f"Data Matrix X shape: {X.shape}")
print(f"  - 784 features (28×28 pixels)")
print(f"  - {N} samples (digit images)")

# ==============================================================================
# PART 4: COMPUTE THE GLOBAL MEAN IMAGE
# ==============================================================================
# WHAT IS THE MEAN IMAGE?
# The mean image μ is the average pixel value across ALL N images
# It represents the "typical" or "prototype" digit for our selected classes
#
# MATHEMATICAL DEFINITION:
# μ = (1/N) × X × 1ᵀ
# where 1 is a column vector of ones
# This formula computes: for each pixel, average its value across all images
#
# In NumPy implementation:
# np.mean(X, axis=1, keepdims=True)
#
# PARAMETER EXPLANATION:
# - axis=1: Compute mean along the COLUMN direction
#   * axis=0 would average down to one value per column (wrong direction)
#   * axis=1 averages across ALL columns (all images) for each row (each pixel)
#   * Result: One average value per pixel, computed across all N images
#
# - keepdims=True: Preserve the 2D tensor structure
#   * Without keepdims: shape would be (784,) → 1D array
#   * With keepdims: shape is (784, 1) → 2D column vector
#   * keepdims=True is ESSENTIAL for proper broadcasting in later operations
#   * Allows μ to be subtracted from every column of X (see Part 5)
#
# SHAPE AND INTERPRETATION:
# μ has shape (784, 1):
#   - 784 rows: One average intensity value per pixel
#   - 1 column: A single "prototype" image

mu = np.mean(X, axis=1, keepdims=True)  # Shape: (784, 1)

print(f"Mean vector mu shape: {mu.shape}")
print(f"  - Represents the average pixel intensity across all images")
print(f"  - Can be reshaped to 28×28 to visualize the 'prototype' digit")

# ==============================================================================
# OPTIONAL: VISUALIZE THE GLOBAL MEAN IMAGE
# ==============================================================================
# PURPOSE: Visual verification that our preprocessing makes sense
# The mean image should look like an "average" or "blurry" version of
# digits 0, 1, and 9 combined
#
# VISUALIZATION STEPS:
# 1. mu.reshape(28, 28): Convert the 784-element vector back to 2D image grid
# 2. plt.imshow(..., cmap='gray'): Display as grayscale image
# 3. Title and formatting: Make the plot clear and interpretable

plt.imshow(mu.reshape(28, 28), cmap='gray')
plt.title("Global Mean Image (Digits 0, 1, 9)")
plt.axis('off')  # Remove axis labels for cleaner visualization
plt.show()

print("\nThe above image shows the 'prototype' digit result from averaging all")

print("the dataset (digits 0, 1, and 9)\n")

# ==============================================================================
# PART 5: FORM THE CENTERED DATA MATRIX
# ==============================================================================
# WHY CENTER THE DATA?
# Centering (subtracting the mean) is a FUNDAMENTAL preprocessing technique for:
#   1. PCA (Principal Component Analysis): Requires zero-centered data
#   2. SVD (Singular Value Decomposition): Works better with centered data
#   3. Regression & Classification: Improves numerical stability
#   4. Statistical Analysis: Centers the data around the origin
#
# MATHEMATICAL DEFINITION:
# X̃ = X - μ × 1ᵀ
# where:
#   - X is the original data matrix (784, N)
#   - μ is the mean vector (784, 1)
#   - 1ᵀ is a row vector of ones (1, N)
#   - The subtraction broadcasts μ to every column of X
#
# RESULT:
# Each image is shifted by subtracting the "average" image from it
# This highlights VARIATIONS/PATTERNS rather than absolute intensity
#
# ==============================================================================
# NUMPY BROADCASTING EXPLANATION
# ==============================================================================
# In NumPy, operations automatically align arrays of different shapes:
#
# X shape:    (784, N)  ← 784 features, N samples
# μ shape:    (784, 1)  ← 784 features, 1 "replicate"
#
# During subtraction (X - μ):
# - NumPy recognizes that μ has only 1 column
# - It automatically replicates (broadcasts) μ across all N columns
# - Equivalent to: X - np.tile(μ, (1, N))
# - Result: Each column of X is independently centered by subtracting μ
#
# VISUAL EXAMPLE (simplified):
# If X = [[100, 110, 105],      and μ = [[105],
#         [200, 220, 210]]                 [210]]
#
# Then X - μ = [[100-105, 110-105, 105-105],    = [[-5,  5,  0],
#                [200-210, 220-210, 210-210]]       [-10, 10, 0]]
#
# IMPORTANCE OF keepdims=True:
# If μ had shape (784,) without the dimension, NumPy wouldn't broadcast correctly
# With keepdims=True, μ shape (784, 1) broadcasts perfectly to subtract from (784, N)

X_tilde = X - mu  # Broadcast subtraction: subtract μ from every column

print(f"Centered Matrix X̃ shape: {X_tilde.shape}")
print(f"  - Same shape as original X")
print(f"  - Each column now represents a CENTERED (zero-mean) image")
print(f"  - Mean of all columns is now approximately zero")

# ==============================================================================
# VERIFICATION OF CENTERING
# ==============================================================================
# After centering, the mean of all columns should be approximately zero
# (small numerical errors due to floating-point precision are expected)

column_means = np.mean(X_tilde, axis=1)
print(f"\nVerification - Mean of centered data (should be ~0): {np.max(np.abs(column_means)):.2e}")
print("Small values (< 1e-6) indicate successful centering")

# ==============================================================================
# SUMMARY OF DATA PREPARATION
# ==============================================================================
print("\n" + "="*80)
print("DATA PREPARATION SUMMARY")
print("="*80)
print(f"✓ Loaded MNIST dataset: {70000} total images")
print(f"✓ Filtered for digits [0, 1, 9]: {N} images selected")
print(f"✓ Vectorized images: {N} vectors of {784} pixels each")
print(f"✓ Data matrix X shape: {X.shape} (features × samples)")
print(f"✓ Mean vector μ shape: {mu.shape}")
print(f"✓ Centered matrix X̃ shape: {X_tilde.shape}")
print("✓ Data is now ready for PCA, SVD, or other ML algorithms")
print("="*80)