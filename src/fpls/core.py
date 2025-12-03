"""
Core Functional Partial Least Squares (FPLS) estimators.

This module implements the conjugate gradient algorithm for functional PLS regression
as described in "Functional Partial Least-Squares: Adaptive Estimation and Inference"
by Babii, Carrasco, and Tsafack (JASA).
"""

from typing import Optional, Tuple, Union
import numpy as np
import pandas as pd


class FunctionalPLS:
    """
    Functional Partial Least Squares estimator using conjugate gradient algorithm.
    
    The estimator solves the equation Kb = r where K is a self-adjoint operator, using
    conjugate gradient iterations. Integrals are approximated with Riemann sums
    over a uniform grid.
    
    Parameters
    ----------
    m_max : int
        Maximum number of PLS components to compute.
    grid_size : Optional[int]
        Number of grid points for functional data discretization.
        If None, will be inferred from X.
    
    Attributes
    ----------
    coef_ : np.ndarray
        Coefficient array of shape (T, m_max+1) where T is the number of
        functional dimensions. Column j contains coefficients using j components.
    n_components_fitted_ : int
        Number of components actually fitted.
    ds_ : float
        Grid spacing for integral approximation.
    """
    
    def __init__(self, m_max: int = 10, grid_size: Optional[int] = None):
        self.m_max = m_max
        self.grid_size = grid_size
        self.coef_ = None
        self.n_components_fitted_ = None
        self.ds_ = None
        
    def fit(
        self, 
        X: Union[np.ndarray, pd.DataFrame], 
        y: Union[np.ndarray, pd.Series],
        ds: Optional[float] = None
    ) -> "FunctionalPLS":
        """
        Fit the Functional PLS model.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data representing functional covariates evaluated on a grid.
        y : array-like of shape (n_samples,)
            Target values.
        ds : float, optional
            Grid spacing for Riemann sum approximation. If None, will be
            computed assuming uniform grid on [0, n_features-1].
            
        Returns
        -------
        self : FunctionalPLS
            Fitted estimator.
        """
        # Convert to numpy arrays
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, (pd.Series, pd.DataFrame)):
            y = y.values
            
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()
        
        n_samples = X.shape[0]
        n_features = X.shape[1]
        
        # Compute grid spacing
        if ds is None:
            if self.grid_size is not None:
                self.ds_ = (n_features - 1) / (self.grid_size - 1)
            else:
                self.ds_ = 1.0
        else:
            self.ds_ = ds
            
        # Prepare data
        y_2d = y.reshape(-1, 1)
        
        # Compute r and K for PLS
        r = X.T @ y_2d / n_samples
        K = X.T @ X * self.ds_ / n_samples
        
        # Run PLS conjugate gradient
        self.coef_ = self._pls_conjugate_gradient(r, K)
        self.n_components_fitted_ = self.m_max
        
        return self
    
    def _pls_conjugate_gradient(
        self, 
        r: np.ndarray, 
        K: np.ndarray
    ) -> np.ndarray:
        """
        Conjugate Gradient algorithm for solving Kb = r with self-adjoint K.
        
        Parameters
        ----------
        r : np.ndarray of shape (n_features, 1)
            Right-hand side vector.
        K : np.ndarray of shape (n_features, n_features)
            Self-adjoint kernel matrix.
            
        Returns
        -------
        βhat : np.ndarray of shape (n_features, m_max+1)
            PLS coefficient estimates. Column 0 is zero, column j gives
            the estimate using j components.
        """
        n_features = len(r)
        βhat = np.zeros((n_features, self.m_max + 1))
        
        # Initialize residual and conjugate direction
        e = r.copy()
        d = r.copy()
        
        for j in range(self.m_max):
            Kd = K @ d
            α1 = e.T @ K @ e
            α = α1 / (Kd.T @ Kd)  # step size for the slope
            
            # Handle both pandas and numpy types
            if hasattr(α, 'values'):
                α_val = α.values[0, 0] if α.ndim > 1 else float(α.values)
            else:
                α_val = float(α)
            
            βhat[:, j+1] = βhat[:, j] + (d * α_val).ravel()  # update the slope
            e = e - Kd * α_val  # update the residual
            
            γ = e.T @ K @ e / α1  # step size for conjugate direction
            
            if hasattr(γ, 'values'):
                γ_val = γ.values[0, 0] if γ.ndim > 1 else float(γ.values)
            else:
                γ_val = float(γ)
                
            d = e + γ_val * d  # update the conjugate vector
            
        return βhat
    
    def predict(
        self, 
        X: Union[np.ndarray, pd.DataFrame], 
        n_components: Optional[int] = None
    ) -> np.ndarray:
        """
        Predict using the Functional PLS model.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict.
        n_components : int, optional
            Number of components to use for prediction. If None, uses all
            fitted components (m_max).
            
        Returns
        -------
        y_pred : np.ndarray of shape (n_samples,)
            Predicted values.
        """
        if self.coef_ is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
            
        # Convert to numpy
        if isinstance(X, pd.DataFrame):
            X = X.values
        X = np.asarray(X, dtype=np.float64)
        
        # Determine number of components
        if n_components is None:
            n_components = self.n_components_fitted_
        elif n_components > self.n_components_fitted_:
            raise ValueError(
                f"n_components={n_components} is greater than the number "
                f"of fitted components ({self.n_components_fitted_})"
            )
            
        # Predict
        β = self.coef_[:, n_components]
        y_pred = (X @ β * self.ds_).ravel()
        
        return y_pred
    
    def fit_predict(
        self, 
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        ds: Optional[float] = None,
        n_components: Optional[int] = None
    ) -> np.ndarray:
        """
        Fit the model and return predictions on the training data.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.
        ds : float, optional
            Grid spacing for integral approximation.
        n_components : int, optional
            Number of components to use for prediction.
            
        Returns
        -------
        y_pred : np.ndarray of shape (n_samples,)
            Predicted values.
        """
        self.fit(X, y, ds=ds)
        return self.predict(X, n_components=n_components)


def fit_fpls(
    X: Union[np.ndarray, pd.DataFrame],
    y: Union[np.ndarray, pd.Series],
    m_max: int = 10,
    ds: Optional[float] = None
) -> Tuple[np.ndarray, float]:
    """
    Convenience function to fit Functional PLS and return coefficients.
    
    This is a functional API that wraps the FunctionalPLS class for
    quick usage similar to the original research code.
    
    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Functional covariate data evaluated on a grid.
    y : array-like of shape (n_samples,)
        Response variable.
    m_max : int, default=10
        Maximum number of PLS components.
    ds : float, optional
        Grid spacing. If None, assumes unit spacing.
        
    Returns
    -------
    coef : np.ndarray of shape (n_features, m_max+1)
        PLS coefficient estimates for 0, 1, ..., m_max components.
    ds : float
        The grid spacing used.
        
    Examples
    --------
    >>> import numpy as np
    >>> from fpls import fit_fpls
    >>> X = np.random.randn(100, 50)  # 100 samples, 50 grid points
    >>> y = X @ np.sin(np.linspace(0, 1, 50)) + np.random.randn(100) * 0.1
    >>> coef, ds = fit_fpls(X, y, m_max=5)
    >>> coef.shape
    (50, 6)
    """
    model = FunctionalPLS(m_max=m_max)
    model.fit(X, y, ds=ds)
    return model.coef_, model.ds_
