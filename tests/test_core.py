"""Tests for the core FunctionalPLS estimator."""

import numpy as np
import pytest
from fpls import FunctionalPLS, fit_fpls


def test_fpls_import():
    """Test that the main class can be imported."""
    assert FunctionalPLS is not None


def test_fpls_initialization():
    """Test FunctionalPLS initialization."""
    model = FunctionalPLS(m_max=5)
    assert model.m_max == 5
    assert model.coef_ is None
    assert model.n_components_fitted_ is None


def test_fpls_fit_basic():
    """Test basic fitting on synthetic data."""
    np.random.seed(42)
    
    # Generate simple synthetic data
    n_samples = 50
    n_features = 20
    
    X = np.random.randn(n_samples, n_features)
    beta_true = np.sin(np.linspace(0, np.pi, n_features))
    y = X @ beta_true + np.random.randn(n_samples) * 0.1
    
    # Fit model
    model = FunctionalPLS(m_max=5)
    model.fit(X, y)
    
    # Check that coefficients were fitted
    assert model.coef_ is not None
    assert model.coef_.shape == (n_features, 6)  # m_max+1 columns
    assert model.n_components_fitted_ == 5
    assert model.ds_ is not None


def test_fpls_predict():
    """Test prediction functionality."""
    np.random.seed(42)
    
    n_samples = 50
    n_features = 20
    
    X = np.random.randn(n_samples, n_features)
    y = np.random.randn(n_samples)
    
    # Fit and predict
    model = FunctionalPLS(m_max=3)
    model.fit(X, y)
    y_pred = model.predict(X, n_components=2)
    
    # Check predictions
    assert y_pred is not None
    assert y_pred.shape == (n_samples,)
    assert np.all(np.isfinite(y_pred))


def test_fpls_fit_predict():
    """Test fit_predict method."""
    np.random.seed(42)
    
    X = np.random.randn(30, 15)
    y = np.random.randn(30)
    
    model = FunctionalPLS(m_max=4)
    y_pred = model.fit_predict(X, y, n_components=3)
    
    assert y_pred.shape == (30,)
    assert model.coef_ is not None


def test_fit_fpls_function():
    """Test the functional API fit_fpls."""
    np.random.seed(42)
    
    X = np.random.randn(40, 25)
    y = np.random.randn(40)
    
    coef, ds = fit_fpls(X, y, m_max=6, ds=0.1)
    
    assert coef.shape == (25, 7)  # m_max+1
    assert ds == 0.1
    assert np.all(np.isfinite(coef))


def test_fpls_with_pandas():
    """Test that the estimator works with pandas DataFrames."""
    np.random.seed(42)
    
    try:
        import pandas as pd
    except ImportError:
        pytest.skip("pandas not installed")
    
    n_samples = 30
    n_features = 10
    
    X_np = np.random.randn(n_samples, n_features)
    y_np = np.random.randn(n_samples)
    
    X_df = pd.DataFrame(X_np)
    y_series = pd.Series(y_np)
    
    # Fit with pandas
    model = FunctionalPLS(m_max=3)
    model.fit(X_df, y_series)
    y_pred = model.predict(X_df)
    
    assert y_pred.shape == (n_samples,)
    assert model.coef_ is not None


def test_fpls_custom_ds():
    """Test fitting with custom grid spacing."""
    np.random.seed(42)
    
    X = np.random.randn(25, 30)
    y = np.random.randn(25)
    
    model = FunctionalPLS(m_max=4)
    model.fit(X, y, ds=0.5)
    
    assert model.ds_ == 0.5


def test_fpls_recovers_signal():
    """Test that FPLS can recover a known signal."""
    np.random.seed(42)
    
    n_samples = 200
    n_features = 50
    
    # Create a clear functional relationship
    grid = np.linspace(0, 1, n_features)
    beta_true = np.exp(-3 * grid)
    
    X = np.random.randn(n_samples, n_features)
    y = X @ beta_true + np.random.randn(n_samples) * 0.05
    
    # Fit with enough components
    model = FunctionalPLS(m_max=10)
    model.fit(X, y, ds=1.0/(n_features-1))
    y_pred = model.predict(X, n_components=10)
    
    # Should predict reasonably well
    r2 = 1 - np.sum((y - y_pred)**2) / np.sum((y - np.mean(y))**2)
    assert r2 > 0.5  # At least moderate fit


def test_fpls_zero_components_raises():
    """Test that requesting zero components raises an error."""
    np.random.seed(42)
    
    X = np.random.randn(20, 10)
    y = np.random.randn(20)
    
    model = FunctionalPLS(m_max=5)
    model.fit(X, y)
    
    # Requesting more components than fitted should raise
    with pytest.raises(ValueError):
        model.predict(X, n_components=10)


def test_fpls_predict_before_fit_raises():
    """Test that predicting before fitting raises an error."""
    model = FunctionalPLS(m_max=5)
    X = np.random.randn(10, 5)
    
    with pytest.raises(ValueError, match="has not been fitted"):
        model.predict(X)
