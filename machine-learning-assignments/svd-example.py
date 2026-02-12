import numpy as np

# Defining a 2x3 matrix
matrix = np.array([[1, 2, 3], [4, 5, 6]])

# Perform SVD
U, s, Vh = np.linalg.svd(matrix)

# Display results
print("U Matrix:\n", U)
print("Singular Values:", s)
print("Vh Matrix:\n", Vh)

# The original matrix can be reconstructed as U @ S_matrix @ Vh
# where S_matrix is a diagonal matrix of s with appropriate padding.
