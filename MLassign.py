import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

x = [2, 1.08, -0.83, -1.97, -1.31, 0.57]
y = [0, 1.68, 1.82, 0.28, -1.51, -1.91]
z = [1, 2.38, 2.49, 2.15, 2.59, 4.32]
t = [1, 2, 3, 4, 5, 6]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot(x, y, z, marker='o')
for i in range(len(t)):
    ax.text(x[i], y[i], z[i], f't={t[i]}')

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_title('Tracked 3D Trajectory')

plt.show()