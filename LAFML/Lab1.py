import numpy as np

## Solving system of linear equations.
A = np.array(
[
    [4, -3, 1],
    [2, 1, 3],
    [-1, 2, -5]
], dtype=np.dtype(float))

b = np.array([-10, 0, 17], dtype=np.dtype(float))

print("Matrix A:")
print(A)
print("\nArray b:")
print(b)

print(f"Shape of A: {np.shape(A)}")
print(f"Shape of b: {np.shape(b)}")

x = np.linalg.solve(A, b)

print(x)
print(f"Solution: {x}")

d = np.linalg.det(A)

print(f"Det A: {d:.2f}")

# Row reduction method

A_system = np.hstack((A, b.reshape((3,1))))

print(A_system)

# exchange row_num of the matrix M with its multiple by row_num_multiple
# Note: for simplicity, you can drop check if  row_num_multiple has non-zero value, which makes the operation valid
def MultiplyRow(M, row_num, row_num_multiple):
    # .copy() function is required here to keep the original matrix without any changes
    M_new = M.copy()
    M_new[row_num] = M_new[row_num] * row_num_multiple
    return M_new

def AddRows(M, row_num_1, row_num_2, row_num_1_multiple):
    M_new = M.copy()
    M_new[row_num_2] = row_num_1_multiple * M_new[row_num_1] + M_new[row_num_2]
    return M_new

def SwapRows(M, row_num_1, row_num_2):
    M_new = M.copy()
    M_new[[row_num_1, row_num_2]] = M_new[[row_num_2, row_num_1]]
    return M_new