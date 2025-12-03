"""
Model selection utilities for Functional PLS.

This module implements adaptive early stopping rules and component selection
as described in Babii, Carrasco, and Tsafack (JASA).
"""

from typing import Optional, Tuple, Union
import numpy as np
import pandas as pd
try:
    import statsmodels.api as sm
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False


def stopping_criterion(
    tau: float,
    sigma: float,
    E: float,
    delta: float,
    n: int
) -> float:
    """
    Compute the right-hand side of the stopping rule.
    
    Parameters
    ----------
    tau : float
        Tuning parameter for the stopping rule (typically > 1).
    sigma : float
        Estimated noise standard deviation.
    E : float
        Estimated second moment E|X|^2.
    delta : float
        Confidence parameter (0 < delta < 1).
    n : int
        Sample size.
        
    Returns
    -------
    threshold : float
        The stopping threshold value.
    """
    return tau * sigma * np.sqrt((2 * E) / (delta * n))


def select_components(
    X: Union[np.ndarray, pd.DataFrame],
    y: Union[np.ndarray, pd.Series],
    m_max: int = 10,
    ds: Optional[float] = None,
    tau: float = 1.01,
    delta: float = 0.1,
    xi: float = 0.01,
    verbose: bool = False
) -> int:
    """
    Select the optimal number of components using early stopping rule.
    
    This function implements the data-driven selection of the number of
    components based on the stopping criterion from the paper.
    
    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Functional covariate data.
    y : array-like of shape (n_samples,)
        Response variable.
    m_max : int, default=10
        Maximum number of components to consider.
    ds : float, optional
        Grid spacing for integral approximation.
    tau : float, default=1.01
        Tuning parameter for stopping rule (>1).
    delta : float, default=0.1
        Confidence parameter (0 < delta < 1).
    xi : float, default=0.01
        Convergence tolerance for sigma estimation.
    verbose : bool, default=False
        If True, print progress information.
        
    Returns
    -------
    m_hat : int
        Selected number of components (between 1 and m_max).
        
    Notes
    -----
    This implements Algorithm 1 from the paper, which adaptively determines
    the number of PLS components to use based on a data-driven stopping rule.
    """
    if not HAS_STATSMODELS:
        raise ImportError(
            "statsmodels is required for component selection. "
            "Install it with: pip install statsmodels"
        )
    
    # Convert to numpy
    if isinstance(X, pd.DataFrame):
        X = X.values
    if isinstance(y, (pd.Series, pd.DataFrame)):
        y = y.values
        
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).ravel()
    
    n, T = X.shape
    
    # Compute grid spacing
    if ds is None:
        ds = 1.0
        
    # Expand y to 2D
    y_2d = y.reshape(-1, 1)
    
    # Compute r and K
    r = X.T @ y_2d / n
    K = X.T @ X * ds / n
    
    # Estimate E|X|^2
    E = np.sum(X**2 * ds) / n
    
    # Initial sigma estimate using OLS
    model_ols = sm.OLS(y_2d, X)
    results_ols = model_ols.fit()
    beta_0 = results_ols.params.reshape(-1, 1)
    
    residuals = y_2d - (X @ beta_0)
    sigma_0_sq = np.sum(residuals**2) / n
    sigma = np.sqrt(sigma_0_sq)
    
    if verbose:
        print(f"Initial sigma estimate: {sigma:.4f}")
    
    # Iteratively refine sigma using PLS
    sigma_prev = sigma
    for iter_sigma in range(10):  # Max 10 iterations for sigma
        # Compute PLS with current sigma
        beta_pls = _pls_for_selection(X, y_2d, r, K, m_max, tau, sigma, E, delta, n, ds)
        
        if beta_pls is not None:
            residuals = y_2d - (X @ beta_pls * ds)
            sigma_sq = np.sum(residuals**2) / n
            sigma = np.sqrt(sigma_sq)
            
            if verbose:
                print(f"Sigma iteration {iter_sigma+1}: {sigma:.4f}")
            
            if np.abs(sigma**2 - sigma_prev**2) < xi:
                break
                
            sigma_prev = sigma
        else:
            break
    
    # Final pass to determine m_hat
    m_hat = _find_stopping_point(X, y_2d, r, K, m_max, tau, sigma, E, delta, n, ds)
    
    if verbose:
        print(f"Selected {m_hat} components")
    
    return m_hat


def _pls_for_selection(
    X: np.ndarray,
    y: np.ndarray,
    r: np.ndarray,
    K: np.ndarray,
    m_max: int,
    tau: float,
    sigma: float,
    E: float,
    delta: float,
    n: int,
    ds: float
) -> Optional[np.ndarray]:
    """Helper function: run PLS and return beta when stopping criterion is met."""
    T = len(r)
    beta_hat = np.zeros((T, m_max + 1))
    e = r.copy()
    d = r.copy()
    
    threshold = stopping_criterion(tau, sigma, E, delta, n)
    
    for j in range(m_max):
        Kd = K @ d
        alpha_1 = e.T @ K @ e
        alpha = alpha_1 / (Kd.T @ Kd)
        
        alpha_val = float(alpha)
        beta_hat[:, j+1] = beta_hat[:, j] + (d * alpha_val).ravel()
        e = e - Kd * alpha_val
        
        gamma = e.T @ K @ e / alpha_1
        gamma_val = float(gamma)
        d = e + gamma_val * d
        
        # Check stopping criterion
        KB = X.T @ (X @ beta_hat[:, j+1].reshape(-1, 1) * ds) / n
        norm_val = np.sqrt(np.sum((r - KB)**2) * ds)
        
        if norm_val <= threshold:
            return beta_hat[:, j+1].reshape(-1, 1)
    
    return None


def _find_stopping_point(
    X: np.ndarray,
    y: np.ndarray,
    r: np.ndarray,
    K: np.ndarray,
    m_max: int,
    tau: float,
    sigma: float,
    E: float,
    delta: float,
    n: int,
    ds: float
) -> int:
    """Helper function: find the component where stopping criterion is satisfied."""
    T = len(r)
    beta_hat = np.zeros((T, m_max + 1))
    e = r.copy()
    d = r.copy()
    
    threshold = stopping_criterion(tau, sigma, E, delta, n)
    norms = []
    
    for j in range(m_max):
        Kd = K @ d
        alpha_1 = e.T @ K @ e
        alpha = alpha_1 / (Kd.T @ Kd)
        
        alpha_val = float(alpha)
        beta_hat[:, j+1] = beta_hat[:, j] + (d * alpha_val).ravel()
        e = e - Kd * alpha_val
        
        gamma = e.T @ K @ e / alpha_1
        gamma_val = float(gamma)
        d = e + gamma_val * d
        
        # Compute norm
        KB = X.T @ (X @ beta_hat[:, j+1].reshape(-1, 1) * ds) / n
        norm_val = np.sqrt(np.sum((r - KB)**2) * ds)
        norms.append(norm_val)
        
        if norm_val <= threshold:
            return j + 1
    
    # If never stopped, return m_max
    return m_max
