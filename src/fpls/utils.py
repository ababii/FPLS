"""
Utility functions for Functional PLS analysis.

This module provides helper functions for preprocessing, grid construction,
and computing common metrics.
"""

from typing import Optional, Tuple, Union
import numpy as np
import pandas as pd


def create_uniform_grid(
    start: float = 0.0,
    end: float = 1.0,
    n_points: int = 100
) -> Tuple[np.ndarray, float]:
    """
    Create a uniform grid for functional data discretization.
    
    Parameters
    ----------
    start : float, default=0.0
        Starting point of the grid.
    end : float, default=1.0
        Ending point of the grid.
    n_points : int, default=100
        Number of grid points.
        
    Returns
    -------
    grid : np.ndarray of shape (n_points,)
        The uniform grid points.
    ds : float
        Grid spacing (step size).
        
    Examples
    --------
    >>> grid, ds = create_uniform_grid(0, 10, 101)
    >>> grid.shape
    (101,)
    >>> ds
    0.1
    """
    grid = np.linspace(start, end, n_points)
    ds = (end - start) / (n_points - 1) if n_points > 1 else 0.0
    return grid, ds


def compute_mse(
    y_true: Union[np.ndarray, pd.Series],
    y_pred: np.ndarray
) -> float:
    """
    Compute mean squared error.
    
    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True target values.
    y_pred : array-like of shape (n_samples,)
        Predicted values.
        
    Returns
    -------
    mse : float
        Mean squared error.
    """
    if isinstance(y_true, (pd.Series, pd.DataFrame)):
        y_true = y_true.values
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    
    return np.mean((y_true - y_pred) ** 2)


def compute_r2(
    y_true: Union[np.ndarray, pd.Series],
    y_pred: np.ndarray
) -> float:
    """
    Compute R-squared (coefficient of determination).
    
    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True target values.
    y_pred : array-like of shape (n_samples,)
        Predicted values.
        
    Returns
    -------
    r2 : float
        R-squared score.
    """
    if isinstance(y_true, (pd.Series, pd.DataFrame)):
        y_true = y_true.values
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    
    return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0


def riemann_integral(
    f: np.ndarray,
    ds: float
) -> float:
    """
    Compute Riemann integral approximation.
    
    Parameters
    ----------
    f : np.ndarray
        Function values on a uniform grid.
    ds : float
        Grid spacing.
        
    Returns
    -------
    integral : float
        Approximate integral value.
        
    Examples
    --------
    >>> # Integrate f(x) = x over [0, 1] with 101 points
    >>> x = np.linspace(0, 1, 101)
    >>> f = x
    >>> ds = 0.01
    >>> riemann_integral(f, ds)  # Should be close to 0.5
    0.5
    """
    return np.sum(f) * ds


def center_functional_data(
    X: Union[np.ndarray, pd.DataFrame]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Center functional data by subtracting the mean function.
    
    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Functional data matrix where each row is a function.
        
    Returns
    -------
    X_centered : np.ndarray of shape (n_samples, n_features)
        Centered functional data.
    mean_function : np.ndarray of shape (n_features,)
        The mean function that was subtracted.
    """
    if isinstance(X, pd.DataFrame):
        X = X.values
    X = np.asarray(X, dtype=np.float64)
    
    mean_function = np.mean(X, axis=0)
    X_centered = X - mean_function
    
    return X_centered, mean_function


def frisch_waugh_residualize(
    Z: Union[np.ndarray, pd.DataFrame],
    X_controls: Union[np.ndarray, pd.DataFrame]
) -> np.ndarray:
    """
    Apply Frisch-Waugh-Lovell theorem to residualize out control variables.
    
    This function regresses Z on X_controls and returns the residuals,
    effectively removing the linear effect of the controls.
    
    Parameters
    ----------
    Z : array-like of shape (n_samples, n_features)
        Variables to be residualized (can be multivariate).
    X_controls : array-like of shape (n_samples, n_controls)
        Control variables.
        
    Returns
    -------
    residuals : np.ndarray
        Residuals after regressing Z on X_controls.
        
    Notes
    -----
    Requires statsmodels package. If Z is multivariate, residuals
    for each column are computed separately.
    """
    try:
        import statsmodels.api as sm
    except ImportError:
        raise ImportError(
            "statsmodels is required for residualization. "
            "Install it with: pip install statsmodels"
        )
    
    # Convert to numpy
    if isinstance(Z, pd.DataFrame):
        Z = Z.values
    if isinstance(X_controls, pd.DataFrame):
        X_controls = X_controls.values
        
    Z = np.asarray(Z, dtype=np.float64)
    X_controls = np.asarray(X_controls, dtype=np.float64)
    
    # Add constant to controls
    X_controls = sm.add_constant(X_controls)
    
    # Fit OLS and get residuals
    model = sm.OLS(Z, X_controls)
    results = model.fit()
    
    return results.resid


def validate_input(
    X: Union[np.ndarray, pd.DataFrame],
    y: Optional[Union[np.ndarray, pd.Series]] = None
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Validate and convert input arrays to numpy format.
    
    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Input features.
    y : array-like of shape (n_samples,), optional
        Target values.
        
    Returns
    -------
    X_validated : np.ndarray
        Validated and converted X.
    y_validated : np.ndarray or None
        Validated and converted y (None if y was None).
        
    Raises
    ------
    ValueError
        If input shapes are inconsistent.
    """
    # Convert X
    if isinstance(X, pd.DataFrame):
        X = X.values
    X = np.asarray(X, dtype=np.float64)
    
    if X.ndim != 2:
        raise ValueError(f"X must be 2-dimensional, got shape {X.shape}")
    
    # Convert y if provided
    if y is not None:
        if isinstance(y, (pd.Series, pd.DataFrame)):
            y = y.values
        y = np.asarray(y, dtype=np.float64).ravel()
        
        if y.ndim != 1:
            raise ValueError(f"y must be 1-dimensional, got shape {y.shape}")
        
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"X and y must have the same number of samples. "
                f"Got X.shape[0]={X.shape[0]} and y.shape[0]={y.shape[0]}"
            )
    
    return X, y
