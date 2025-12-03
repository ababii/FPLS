"""
Basic usage example for the fpls package.

This script demonstrates how to use the FunctionalPLS estimator
on synthetic data.
"""

import numpy as np
from fpls import FunctionalPLS, fit_fpls, compute_r2, compute_mse

# Set random seed for reproducibility
np.random.seed(42)

# Generate synthetic functional data
n = 1000  # sample size
J = 100  # number of basis functions
T = 200  # number of grid points
m = 10   # number of components for model

# Grid points
s = np.linspace(0, 1, T)
ds = s[1] - s[0]

# Generate eigenvalues
j = np.arange(1, J + 1)
λ = 2 / (j ** 1.1)  # eigenvalues

# Generate eigenfunctions (Fourier basis) - vectorized
v = np.sqrt(2) * np.cos(np.pi * s[:, np.newaxis] * j[np.newaxis, :])
v[:, 0] = 1  # first eigenfunction is constant

# Generate coefficient function β
beta_true = v @ (4 / (j ** 2.7))

# Generate functional covariate X
X = np.random.randn(n, J) * np.sqrt(λ) @ v.T

# Generate response
y = X @ beta_true / T + np.random.randn(n)

# Split into train/test
n_train = 750
X_train, X_test = X[:n_train], X[n_train:]
y_train, y_test = y[:n_train], y[n_train:]

print("=" * 60)
print("FPLS Basic Usage Example")
print("=" * 60)

# Example 1: Using the class API
print("\n1. Using FunctionalPLS class API:")
print("-" * 40)

model = FunctionalPLS(m_max=m)
model.fit(X_train, y_train, ds=ds)

# Predict with different numbers of components
for m_comp in [1, 2, 3, 4, 5]:
    y_pred = model.predict(X_test, n_components=m_comp)
    r2 = compute_r2(y_test, y_pred)
    mse = compute_mse(y_test, y_pred)
    print(f"  m={m_comp:2d}: R² = {r2:.4f}, MSE = {mse:.4f}")

# Example 2: Using the functional API
print("\n2. Using functional API (fit_fpls):")
print("-" * 40)

coef, ds_fitted = fit_fpls(X_train, y_train, m_max=m, ds=ds)
print(f"  Coefficient array shape: {coef.shape}")
print(f"  Grid spacing: {ds_fitted}")

# Use coefficients for prediction
m_comp = 3
y_pred = (X_test @ coef[:, m_comp] * ds)
r2 = compute_r2(y_test, y_pred)
print(f"  Using {m_comp} components: R² = {r2:.4f}")

# Example 3: Comparing with true coefficients
print("\n3. Coefficient recovery:")
print("-" * 40)

import matplotlib.pyplot as plt
beta_pls = coef[:, m_comp]

plt.figure(figsize=(10, 6))
plt.plot(s, beta_true, 'b-', linewidth=2, label='True β (normalized)', alpha=0.7)
plt.plot(s, beta_pls, 'r--', linewidth=2, label='Estimated β (normalized)', alpha=0.7)
plt.xlabel('Grid points (s)', fontsize=12)
plt.ylabel('Coefficient value', fontsize=12)
plt.title('Comparison of True and Estimated Coefficient Functions', fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print("\n" + "=" * 60)
print("Example completed successfully!")
print("=" * 60)
