"""
Functional Partial Least Squares (FPLS) Package.

A Python package for functional regression using partial least squares,
implementing the methods from:

    Babii, A., Carrasco, M., & Tsafack, I. (2023).
    "Functional Partial Least-Squares: Adaptive Estimation and Inference"
    Journal of the American Statistical Association.

Main classes and functions:
    - FunctionalPLS: Main estimator class with scikit-learn style API
    - fit_fpls: Convenience function for quick fitting
    - select_components: Adaptive component selection via early stopping
    - plot_coefficient_function: Visualize FPLS coefficient functions
    - plot_comparison: Compare multiple coefficient functions
    - load_example_data: Load example crop yield datasets

Example:
    >>> from fpls import FunctionalPLS
    >>> import numpy as np
    >>> X = np.random.randn(100, 50)
    >>> y = np.random.randn(100)
    >>> model = FunctionalPLS(m_max=10)
    >>> model.fit(X, y)
    >>> y_pred = model.predict(X)
"""

__version__ = "0.1.0"
__author__ = "Andrii Babii, Marine Carrasco, Idriss Tsafack"

from .core import FunctionalPLS, fit_fpls
from .selection import select_components, stopping_criterion
from .utils import (
    create_uniform_grid,
    compute_mse,
    compute_r2,
    center_functional_data,
    frisch_waugh_residualize,
    validate_input,
)
from .visualization import (
    plot_coefficient_function,
    plot_comparison,
    load_example_data,
)

__all__ = [
    "FunctionalPLS",
    "fit_fpls",
    "select_components",
    "stopping_criterion",
    "create_uniform_grid",
    "compute_mse",
    "compute_r2",
    "center_functional_data",
    "frisch_waugh_residualize",
    "validate_input",
    "plot_coefficient_function",
    "plot_comparison",
    "load_example_data",
    "__version__",
]
