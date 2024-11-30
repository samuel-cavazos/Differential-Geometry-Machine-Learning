import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Generate circular data
np.random.seed(42)
n_samples = 500
radius = 1.0

# Generate random points in 2D space
X = np.random.uniform(-1.5, 1.5, (n_samples, 2))

# Assign class based on distance from origin
y = np.array([1 if np.linalg.norm(point) < radius else 0 for point in X])

# 2D Visualization
fig, ax = plt.subplots(figsize=(6, 6))
for point, label in zip(X, y):
    color = 'blue' if label == 1 else 'red'
    ax.scatter(point[0], point[1], color=color, s=10, alpha=0.7)
circle = plt.Circle((0, 0), radius, color='black', fill=False, linestyle='--')
ax.add_artist(circle)
ax.axhline(0, color='black', linewidth=0.5)
ax.axvline(0, color='black', linewidth=0.5)
ax.set_title("Circular Classification in 2D Space")
ax.set_xlabel("x1")
ax.set_ylabel("x2")
ax.set_aspect('equal', adjustable='datalim')
plt.grid()
plt.show()

# Map data to a higher-dimensional space: z = x1^2 + x2^2
z = np.sum(X**2, axis=1).reshape(-1, 1)
X_3D = np.hstack((X, z))

# 3D Visualization
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, projection='3d')
for point, label in zip(X_3D, y):
    color = 'blue' if label == 1 else 'red'
    ax.scatter(point[0], point[1], point[2], color=color, s=10, alpha=0.7)
ax.set_title("Circular Classification in 3D Space (Linear Separation)")
ax.set_xlabel("x1")
ax.set_ylabel("x2")
ax.set_zlabel("z = x1^2 + x2^2")

# Add a separating plane
xx, yy = np.linspace(-1.5, 1.5, 10), np.linspace(-1.5, 1.5, 10)
XX, YY = np.meshgrid(xx, yy)
ZZ = radius**2 * np.ones_like(XX)
ax.plot_surface(XX, YY, ZZ, alpha=0.5, color='green')
plt.show()
