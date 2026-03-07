import numpy as np
import matplotlib.pyplot as plt
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

# design matrix for quadratic model
X = np.array([
    [1, 1, 1**2],
    [1, 2, 2**2],
    [1, 3, 3**2],
    [1, 4, 4**2],
    [1, 5, 5**2],
    [1, 6, 6**2]
], dtype=float)

# gradient descent
B = np.zeros((3, 3))   # row 1: constants, row 2: t coefficients, row 3: t^2 coefficients
lr = 0.0002
tol = 1e-12
max_iter = 500000

for _ in range(max_iter):
    grad = -2 * X.T @ (Y - X @ B)
    B_new = B - lr * grad
    if np.linalg.norm(B_new - B) < tol:
        B = B_new
        break
    B = B_new

# results
c0 = B[0]
c1 = B[1]
c2 = B[2]

Y_hat = X @ B
sse = np.sum((Y - Y_hat)**2)
acc = 2 * c2

print("x(t) =", c0[0], "+", c1[0], "* t +", c2[0], "* t^2")
print("y(t) =", c0[1], "+", c1[1], "* t +", c2[1], "* t^2")
print("z(t) =", c0[2], "+", c1[2], "* t +", c2[2], "* t^2")

print("Acceleration vector =", acc)
print("SSE =", sse)

# predicted point at t = 7 from part (b)
t_next = 7
p_next = np.array([1, t_next, t_next**2]) @ B

print("Predicted position at t=7:", p_next)

# append predicted point to observed data
Y_all = np.vstack((Y, p_next))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# plot observed points
ax.scatter(Y[:,0], Y[:,1], Y[:,2], label="Observed points")

# plot predicted point
ax.scatter(p_next[0], p_next[1], p_next[2], s=100, marker='x', label="Predicted point at t=7")

# connect all points in time order: t=1,...,6,7
ax.plot(Y_all[:,0], Y_all[:,1], Y_all[:,2], label="Trajectory")

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
ax.legend()
plt.show()