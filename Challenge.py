# %%
import numpy as np
# %%
# Set random seed
np.random.seed(seed=1234)
# %%
# Array creation and properties
x = np.array(6)
print("x : ", x)
print("X ndim : ", x.ndim)
print("X shape : ", x.shape)
print("X datatype : ", x.dtype)
print("X size : ", x.size)
print("\n")
# %%
x = np.array([1, 2, 3])
print("x : ", x)
print("X ndim : ", x.ndim)
print("X shape : ", x.shape)
print("X datatype : ", x.dtype)
print("X size : ", x.size)
# %%
# 2D array
x = np.array([[1, 2, 3], [4, 5, 6]])
print("\nx : ", x)
print("X ndim : ", x.ndim)
print("X shape : ", x.shape)
print("X datatype : ", x.dtype)
print("X size : ", x.size)
# %%
# 3D array
x = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
print("\nx:\n", x)
print("x ndim: ", x.ndim)
print("x shape:", x.shape)
print("x size: ", x.size)
print("x dtype: ", x.dtype)
# %%

# Array creation functions
print("\nnp.zeros((2,2)):\n", np.zeros((2, 2)))
print("np.zeros((3)):\n", np.zeros((3)))
print("np.ones((2,2))\n", np.ones((2, 2)))
print("np.eye((2))\n", np.eye(2))
print("np.random.random((2,2)):\n", np.random.random((2, 2)))
# %%

# Array indexing and slicing
x = np.array([1, 2, 3])
print("\nx: ", x)
print("x[0]: ", x[0])
x[0] = 0
print("x: ", x)
# %%

x = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
print("\n", x)
print("x column 1: ", x[:, 1])
print("x row 0: ", x[0, :])
print("x rows 0,1 & cols 1,2: \n", x[0:2, 1:3])
# %%

# Advanced indexing
rows_to_get = np.array([0, 1, 2])
cols_to_get = np.array([0, 2, 1])
print("\nindexed values: ", x[rows_to_get, cols_to_get])
# %%

# Boolean indexing
x = np.array([[1, 2], [3, 4], [5, 6]])
print("\nx:\n", x)
print("x > 2:\n", x > 2)
print("x[x > 2]:\n", x[x > 2].reshape(2, 2))
# %%

# Array operations
x = np.array([[1, 2], [3, 4]], dtype=np.float64)
y = np.array([[1, 2], [3, 4]], dtype=np.float64)
print("\nx + y:\n", np.add(x, y))
print("x - y:\n", np.subtract(x, y))
print("x * y:\n", np.multiply(x, y))
print("Dot product of X and Y : ", x.dot(y))
# %%

# Array statistics
x = np.array([[1, 2], [3, 4]])
print("\nx:\n", x)
print("sum all : ", np.sum(x))
print("sum along axis 0 : ", np.sum(x, axis=0))
print("sum along axis 1 : ", np.sum(x, axis=1))
print("min : {} and max : {}".format(np.min(x), np.max(x)))
# %%

# Broadcasting
x = np.array([[1, 2], [3, 4]])
y = np.array(3)
z = x + y
print("\nz:\n", z)
# %%

# Reshaping and transposing
a = np.array((3, 4, 5))
b = np.expand_dims(a, axis=1)
c = a + b
print("\nc:\n", c)

# %%
a = a.reshape(-1, 1)
c = a + b
print("\nc:\n", c)
# %%

x = np.array([1, 2, 3, 4, 5, 6])
x = x.reshape(6, -1)
print("\nx:\n", x)
# %%

# Complex reshaping and transposing
x = np.array([[[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3]],
              [[10, 10, 10, 10], [20, 20, 20, 20], [30, 30, 30, 30]]])
print("\nx:\n", x)
print("x.shape: ", x.shape)
y = np.transpose(x, (1, 0, 2))
print("\ny:\n", y)
print("y.shape: ", y.shape)
z_correct = np.reshape(y, (y.shape[0], -1))
print("\nz_correct:\n", z_correct)
print("z_correct.shape: ", z_correct.shape)
# %%

# Concatenation
x = np.random.random((2, 3))
print("\nx:\n", x)
print("x.shape:", x.shape)
y = np.concatenate([x,x], axis=0)
print("\ny:\n", y)
print("y.shape:", y.shape)
# %%

x = np.random.random((2, 3))
print("\nx:\n", x)
print("x.shape:", x.shape)
y = np.concatenate([x,x], axis=1)
print("\ny:\n", y)
print("y.shape:", y.shape)

# %%
# Splitting
x = np.array([[[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3]],
              [[10, 10, 10, 10], [20, 20, 20, 20], [30, 30, 30, 30]]])
x, y = np.vsplit(x, 2)
print("\nx:\n", x)
print("\ny:\n", y)
