import numpy as np

# data
t = np.array([1, 2, 3, 4, 5, 6], dtype=float)
Y = np.array([
    [ 2.00,  0.00, 1.00],
    [ 1.08,  1.68, 2.38],
    [-0.83,  1.82, 2.49],
    [-1.97,  0.28, 2.15],
    [-1.31, -1.51, 2.59],
    [ 0.57, -1.91, 4.32]
], dtype=float)

# design matrix for constant velocity model
X = np.column_stack((np.ones_like(t), t))   # [1, t]

# gradient descent
B = np.zeros((2, 3))   # row 1 = intercepts, row 2 = velocities
lr = 0.001
tol = 1e-12
max_iter = 200000

for _ in range(max_iter):
    grad = -2 * X.T @ (Y - X @ B)
    B_new = B - lr * grad
    if np.linalg.norm(B_new - B) < tol:
        B = B_new
        break
    B = B_new

# results
a = B[0]          # intercepts
v = B[1]          # velocity components
Y_hat = X @ B
sse = np.sum((Y - Y_hat)**2)
speed = np.linalg.norm(v)

print("x(t) =", a[0], "+", v[0], "* t")
print("y(t) =", a[1], "+", v[1], "* t")
print("z(t) =", a[2], "+", v[2], "* t")
print("Velocity vector =", v)
print("Speed =", speed)
print("SSE =", sse)