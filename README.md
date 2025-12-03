# FPLS: Functional Partial Least Squares

A Python package for functional regression using Partial Least Squares (PLS), implementing the methods from:

> **Babii, A., Carrasco, M., & Tsafack, I.** (2023). "Functional Partial Least-Squares: Adaptive Estimation and Inference." *Journal of the American Statistical Association*.

## Overview

This package provides a clean, scikit-learn-style API for fitting functional PLS regression models with adaptive component selection. The core algorithm uses a conjugate gradient method to solve the functional PLS problem, with integrals approximated via Riemann sums on uniform grids.

**Key Features:**
- **FunctionalPLS estimator** with `.fit()`, `.predict()`, and `.fit_predict()` methods
- **Adaptive early stopping** for automatic component selection
- **Efficient conjugate gradient** algorithm for functional data
- Support for both **NumPy arrays** and **Pandas DataFrames**

## Installation

Install in development (editable) mode:

```bash
cd fpls
pip install -e .
```

To include optional dependencies for model selection and plotting:

```bash
pip install -e ".[all]"
```

## Quick Start

### Basic Usage

```python
import numpy as np
from fpls import FunctionalPLS

# Generate synthetic functional data
n_samples = 200
n_grid_points = 50

X = np.random.randn(n_samples, n_grid_points)
beta_true = np.sin(np.linspace(0, 2*np.pi, n_grid_points))
y = X @ beta_true + np.random.randn(n_samples) * 0.5

# Fit Functional PLS
model = FunctionalPLS(m_max=10)
model.fit(X, y, ds=1.0)

# Predict
y_pred = model.predict(X, n_components=5)

# Evaluate
from fpls import compute_r2
r2 = compute_r2(y, y_pred)
print(f"R² = {r2:.3f}")
```

### Adaptive Component Selection

```python
from fpls import select_components

# Automatically select the optimal number of components
m_hat = select_components(X, y, m_max=10, tau=1.01, delta=0.1)
print(f"Selected {m_hat} components")

# Fit with selected components
model = FunctionalPLS(m_max=m_hat)
model.fit(X, y)
y_pred = model.predict(X)
```

### Functional API (Research-Style)

For users familiar with the original replication code:

```python
from fpls import fit_fpls

# Directly get coefficient array
coef, ds = fit_fpls(X, y, m_max=10, ds=1.0)

# coef has shape (n_features, m_max+1)
# coef[:, j] gives coefficients using j components
```

## API Reference

### Main Classes

**`FunctionalPLS(m_max=10, grid_size=None)`**
- `.fit(X, y, ds=None)` - Fit the model
- `.predict(X, n_components=None)` - Make predictions
- `.fit_predict(X, y, ds=None, n_components=None)` - Fit and predict
- `.coef_` - Fitted coefficients (shape: `n_features × (m_max+1)`)

### Functions

**`fit_fpls(X, y, m_max=10, ds=None)`**
- Convenience function returning `(coef, ds)` tuple

**`select_components(X, y, m_max=10, ds=None, tau=1.01, delta=0.1, xi=0.01)`**
- Adaptive early stopping to select optimal number of components

**Utilities:**
- `create_uniform_grid(start, end, n_points)` - Create discretization grid
- `compute_mse(y_true, y_pred)` - Mean squared error
- `compute_r2(y_true, y_pred)` - R-squared
- `center_functional_data(X)` - Center by mean function
- `frisch_waugh_residualize(Z, X_controls)` - Residualize out controls

## Examples

See the `examples/` directory for complete worked examples, including:
- Basic functional regression
- Component selection
- Comparison with standard methods

## Development

Run tests:
```bash
pytest
```

Run tests with coverage:
```bash
pytest --cov=fpls --cov-report=html
```

## Citation

If you use this package in your research, please cite:

```bibtex
@article{babii2023functional,
  title={Functional Partial Least-Squares: Adaptive Estimation and Inference},
  author={Babii, Andrii and Carrasco, Marine and Tsafack, Idriss},
  journal={Journal of the American Statistical Association},
  year={2023}
}
```

## License

MIT License - see LICENSE file for details.

## Authors

- Andrii Babii
- Marine Carrasco
- Idriss Tsafack
