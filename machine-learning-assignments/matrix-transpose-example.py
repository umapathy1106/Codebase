import numpy as np

# Define a 2x3 matrix
matrix = np.array([[1, 2, 3], [4, 5, 6]]) 
# Transpose the matrix
#transposed_matrix = matrix.T  
transposed_matrix = np.transpose(matrix)  # Using the function from the assignment
print("Original Matrix:\n", matrix)
print("Transposed Matrix:\n", transposed_matrix)


transposed_matrix1 = np.matrix.transpose(matrix)
print("Transposed Matrix using np.matrix.transpose:\n", transposed_matrix1)