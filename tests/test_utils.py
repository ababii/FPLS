"""Tests for utility functions."""

import numpy as np
import pytest
from fpls.utils import (
    create_uniform_grid,
    compute_mse,
    compute_r2,
    center_functional_data,
    riemann_integral,
    validate_input,
)


def test_create_uniform_grid():
    """Test grid creation."""
    grid, ds = create_uniform_grid(0, 1, 11)
    
    assert len(grid) == 11
    assert grid[0] == 0.0
    assert grid[-1] == 1.0
    assert ds == 0.1


def test_compute_mse():
    """Test MSE computation."""
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.1, 2.1, 2.9, 4.1])
    
    mse = compute_mse(y_true, y_pred)
    expected_mse = np.mean([0.01, 0.01, 0.01, 0.01])
    
    assert np.isclose(mse, expected_mse)


def test_compute_r2():
    """Test R² computation."""
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = y_true.copy()  # Perfect prediction
    
    r2 = compute_r2(y_true, y_pred)
    assert np.isclose(r2, 1.0)
    
    # Worst prediction (constant mean)
    y_pred_bad = np.ones_like(y_true) * np.mean(y_true)
    r2_bad = compute_r2(y_true, y_pred_bad)
    assert np.isclose(r2_bad, 0.0)


def test_riemann_integral():
    """Test Riemann integral approximation."""
    # Integrate f(x) = x over [0, 1]
    n = 1001
    x = np.linspace(0, 1, n)
    f = x
    ds = 1.0 / (n - 1)
    
    integral = riemann_integral(f, ds)
    assert np.isclose(integral, 0.5, atol=1e-3)


def test_center_functional_data():
    """Test centering of functional data."""
    np.random.seed(42)
    
    X = np.random.randn(50, 20) + 5.0  # Add constant offset
    X_centered, mean_func = center_functional_data(X)
    
    assert X_centered.shape == X.shape
    assert mean_func.shape == (20,)
    assert np.allclose(np.mean(X_centered, axis=0), 0.0, atol=1e-10)


def test_validate_input():
    """Test input validation."""
    X = np.random.randn(30, 10)
    y = np.random.randn(30)
    
    X_val, y_val = validate_input(X, y)
    
    assert X_val.shape == (30, 10)
    assert y_val.shape == (30,)
    
    # Test with mismatched shapes
    y_bad = np.random.randn(20)
    with pytest.raises(ValueError, match="same number of samples"):
        validate_input(X, y_bad)


def test_validate_input_pandas():
    """Test validation with pandas inputs."""
    try:
        import pandas as pd
    except ImportError:
        pytest.skip("pandas not installed")
    
    X_df = pd.DataFrame(np.random.randn(20, 5))
    y_series = pd.Series(np.random.randn(20))
    
    X_val, y_val = validate_input(X_df, y_series)
    
    assert isinstance(X_val, np.ndarray)
    assert isinstance(y_val, np.ndarray)
    assert X_val.shape == (20, 5)
    assert y_val.shape == (20,)
