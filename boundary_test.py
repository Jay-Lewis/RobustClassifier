from optimal_boundary import optimal_boundary
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


# -----------------------------------------------------------------
# Plot different decision boundaries
# -----------------------------------------

ax = plt.axes()

# Get Optimal Boundary (l2) + pts
optimal_boundary_pts, data = optimal_boundary(plot_flag=False, power=2, metric='minkowski')

# Plot Optimal (l2) Boundary
x, y = optimal_boundary_pts.T
ax.plot(x, y, c='g')

# Get Optimal Boundary (linf) + pts
optimal_boundary_pts, _ = optimal_boundary(plot_flag=False, power=1, metric='chebyshev')

# Plot Optimal (linf) Boundary
x, y = optimal_boundary_pts.T
ax.plot(x, y, c='m')

# Plot data points
cm_bright = ListedColormap(['#FF0000', '#0000FF'])
X_train, X_test, y_train, y_test = data
ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright,
           edgecolors='k')
ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6,
           edgecolors='k')


plt.tight_layout()
plt.show()





