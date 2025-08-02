import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

# 1. Generate non-uniform sample data
np.random.seed(42)
n_points = 12
x = np.sort(np.random.uniform(0, 10, n_points))
y = np.exp(-0.1*x) * np.sin(x)

# 2. Manual natural cubic spline implementation
def natural_cubic_spline(x, y, xx):
    n = len(x)
    h = np.diff(x)

    # Step 1: Set up the tridiagonal system
    alpha = np.zeros(n)
    for i in range(1, n-1):
        alpha[i] = (3/h[i])*(y[i+1] - y[i]) - (3/h[i-1])*(y[i] - y[i-1])
    
    # Step 2: Solve tridiagonal system for c (second derivatives)
    l = np.ones(n)
    mu = np.zeros(n)
    z = np.zeros(n)
    c = np.zeros(n)
    b = np.zeros(n-1)
    d = np.zeros(n-1)
    a = y.copy()
    
    for i in range(1, n-1):
        l[i] = 2*(x[i+1] - x[i-1]) - h[i-1]*mu[i-1]
        mu[i] = h[i]/l[i]
        z[i] = (alpha[i] - h[i-1]*z[i-1])/l[i]
    
    # Natural spline boundary conditions
    l[-1] = 1
    z[-1] = 0
    c[-1] = 0
    # Back substitution
    for j in range(n-2, -1, -1):
        c[j] = z[j] - mu[j]*c[j+1]
        b[j] = (a[j+1] - a[j])/h[j] - h[j]*(c[j+1] + 2*c[j])/3
        d[j] = (c[j+1] - c[j])/(3*h[j])
        
    # Step 3: Evaluate spline at desired xx
    spline_vals = np.ones_like(xx)
    for i in range(n-1):
        if i==0: idx = (xx <= x[i+1])
        elif i==n-2: idx = (xx >= x[i])
        else: idx = (xx >= x[i]) & (xx <= x[i+1])
        dx = xx[idx] - x[i]
        spline_vals[idx] = a[i] + b[i]*dx + c[i]*dx**2 + d[i]*dx**3
    return spline_vals

# 3. Create dense evaluation points
xx = np.linspace(0, 10, 400)
true_y = np.exp(-0.1*xx) * np.sin(xx)

# 4. Interpolate using manual implementation
manual_spline = natural_cubic_spline(x, y, xx)

# 5. Interpolate using scipy's CubicSpline
scipy_spline = CubicSpline(x, y, bc_type='natural')
scipy_y = scipy_spline(xx)

# 7. Calculate Mean Absolute Error for both splines
mae_manual = np.mean(np.abs(manual_spline - true_y))
mae_scipy = np.mean(np.abs(scipy_y - true_y))

print(f"Mean Absolute Error (Manual Spline): {mae_manual:.6e}")
print(f"Mean Absolute Error (scipy CubicSpline): {mae_scipy:.6e}")

# 6. Plot results
plt.figure(figsize=(10,5))
plt.plot(xx, true_y, 'k--', label='True $e^{-x}sin(x)$')
plt.plot(x, y, 'o', label='Sample points')
plt.plot(xx, manual_spline, 'b.-', lw=2, label='Manual cubic spline')
plt.plot(xx, scipy_y, 'r.-', lw=2, label='scipy CubicSpline')
plt.legend()
plt.title("Manual vs. scipy Cubic Spline Interpolation")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.tight_layout()
plt.show()