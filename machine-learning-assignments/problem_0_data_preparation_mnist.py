import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist #type: ignore

# 1. Load the MNIST dataset
# We combine train and test sets to maximize N as implied by "total number of selected images"
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_all = np.concatenate((x_train, x_test), axis=0)
y_all = np.concatenate((y_train, y_test), axis=0)

# 2. Extract only digits 0, 1, and 9 
# Create a mask for the selected labels
mask = np.isin(y_all, [0, 1, 9])
x_selected = x_all[mask]
y_selected = y_all[mask]

# N is the total number of selected images [cite: 24]
N = x_selected.shape[0]
print(f"Total number of selected images (N): {N}")

# 3. Vectorize and Stack 
# Reshape each image (28x28) into a vector (784)
# Transpose to stack them as columns: shape becomes (784, N)
# Set data type to float32 [cite: 24]
X = x_selected.reshape(N, 784).T.astype(np.float32)

print(f"Data Matrix X shape: {X.shape}")

# 4. Compute Global Mean Image [cite: 25, 28]
# mu = (1/N) * X * 1 (mathematically equivalent to mean across columns)
mu = np.mean(X, axis=1, keepdims=True)

print(f"Mean vector mu shape: {mu.shape}")

# Optional: Visualize the global mean image to verify
plt.imshow(mu.reshape(28, 28), cmap='gray')
plt.title("Global Mean Image (Digits 0, 1, 9)")
plt.axis('off')
plt.show()

# 5. Form the Centered Data Matrix [cite: 27, 29]
# X_tilde = X - mu * 1_transpose
# NumPy broadcasting handles the subtraction of the column vector mu from every column of X
X_tilde = X - mu

print(f"Centered Matrix X_tilde shape: {X_tilde.shape}")