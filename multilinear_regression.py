import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

X_train = np.array([
    [35, 64],
    [86, 42],
    [18, 72],
    [93, 52],
    [89, 135],
    [172, 63],
    [252, 189],
    [215, 335],
    [164, 186],
    [242, 19]
])
y_train = np.array([63, 74, 51, 42, 85, 64, 34, 26, 75, 54])

def cost_derivative(X, y, w, b):
    m, n = X.shape
    dJ_dw = np.zeros(n)
    dJ_db = 0
    for i in range(m):
        err = (np.dot(X[i], w) + b) - y[i]
        dJ_dw += (1/m) * (err * X[i])
        dJ_db += (1/m) * err
    return dJ_dw, dJ_db

def gradient_descent(n_it, X, y, alpha=0.001):
    m, n = X.shape
    w = np.zeros(n)
    b = 0
    for _ in range(n_it):
        dJ_dw, dJ_db = cost_derivative(X, y, w, b)
        w -= alpha * dJ_dw
        b -= alpha * dJ_db
    return w, b

# Plane Equation
def plane(w, b, X):
    return np.dot(X, w) + b

w, b = gradient_descent(100, X_train, y_train)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(X_train[:, 0], X_train[:, 1], y_train, marker='x', color='red')

x0 = np.linspace(min(X_train[:, 0]), max(X_train[:, 0]), 2)
x1 = np.linspace(min(X_train[:, 1]), max(X_train[:, 1]), 2)
X, Y = np.meshgrid(x0, x1)

Z = plane(w, b, np.array([X.ravel(), Y.ravel()]).T).reshape(X.shape)

ax.plot_surface(X, Y, Z, color='cyan', alpha=0.5, edgecolor='k')

ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
ax.set_zlabel('Target (y)')

plt.show()
