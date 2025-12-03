"""
Visualization utilities for Functional PLS.

This module provides plotting functions for visualizing FPLS coefficient functions
and comparing results across different datasets or time periods.
"""

import numpy as np
import matplotlib.pyplot as plt

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False


def plot_coefficient_function(
    s,
    beta,
    title="FPLS Coefficient Function",
    xlabel="Grid Points",
    ylabel="Coefficient",
    color="#2E86AB",
    figsize=(10, 6),
    fill_between=True,
    style="seaborn",
    ax=None,
    **kwargs
):
    """
    Plot a single FPLS coefficient function.

    Parameters
    ----------
    s : array-like, shape (n_grid_points,)
        Grid points for the functional domain.
    beta : array-like, shape (n_grid_points,)
        Coefficient values at each grid point.
    title : str, optional
        Plot title.
    xlabel : str, optional
        Label for x-axis.
    ylabel : str, optional
        Label for y-axis.
    color : str, optional
        Color for the line plot.
    figsize : tuple, optional
        Figure size (width, height) in inches.
    fill_between : bool, optional
        Whether to fill area under the curve.
    style : str, optional
        Plot style ('seaborn' or 'matplotlib').
    ax : matplotlib.axes.Axes, optional
        Existing axes to plot on. If None, creates new figure.
    **kwargs : dict
        Additional keyword arguments passed to plt.plot().

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object.
    ax : matplotlib.axes.Axes
        The axes object.

    Example
    -------
    >>> from fpls import fit_fpls, plot_coefficient_function
    >>> import numpy as np
    >>> X = np.random.randn(100, 50)
    >>> y = np.random.randn(100)
    >>> coef, ds = fit_fpls(X, y, m_max=5)
    >>> s = np.linspace(0, 1, 50)
    >>> fig, ax = plot_coefficient_function(s, coef[:, 3], title="Component 3")
    >>> plt.show()
    """
    if style == "seaborn" and HAS_SEABORN:
        sns.set_style("ticks")
        sns.set_palette("husl")

    # Create new figure if ax is not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=100)
    else:
        fig = ax.get_figure()

    # Default plot kwargs
    plot_kwargs = {
        "linewidth": 3,
        "alpha": 0.9,
        "label": kwargs.pop("label", "Functional Slope Coefficient"),
    }
    plot_kwargs.update(kwargs)

    # Plot coefficient function
    ax.plot(s, beta, color=color, **plot_kwargs)

    # Fill between if requested
    if fill_between:
        ax.fill_between(s, beta, alpha=0.2, color=color)

    # Styling
    ax.set_xlabel(xlabel, fontsize=14, fontweight="600")
    ax.set_ylabel(ylabel, fontsize=14, fontweight="600")
    ax.set_title(title, fontsize=16, fontweight="bold", pad=20)

    # Legend
    if plot_kwargs.get("label"):
        ax.legend(
            fontsize=12,
            frameon=True,
            shadow=True,
            loc="best",
            edgecolor="gray",
            fancybox=True,
        )

    # Grid
    ax.grid(True, alpha=0.25, linestyle=":", linewidth=1)
    ax.tick_params(labelsize=11)

    # Despine if seaborn is available
    if style == "seaborn" and HAS_SEABORN:
        sns.despine(offset=10, trim=True, ax=ax)

    plt.tight_layout()

    return fig, ax


def plot_comparison(
    s_list,
    beta_list,
    titles,
    colors=None,
    suptitle=None,
    xlabel="Grid Points",
    ylabel="Coefficient",
    figsize=(16, 6),
    fill_between=True,
    style="seaborn",
    **kwargs
):
    """
    Plot multiple FPLS coefficient functions side-by-side for comparison.

    Parameters
    ----------
    s_list : list of array-like
        List of grid points for each coefficient function.
    beta_list : list of array-like
        List of coefficient values for each function.
    titles : list of str
        List of subplot titles.
    colors : list of str, optional
        List of colors for each subplot. If None, uses default palette.
    suptitle : str, optional
        Overall figure title.
    xlabel : str, optional
        Label for x-axis.
    ylabel : str, optional
        Label for y-axis.
    figsize : tuple, optional
        Figure size (width, height) in inches.
    fill_between : bool, optional
        Whether to fill area under curves.
    style : str, optional
        Plot style ('seaborn' or 'matplotlib').
    **kwargs : dict
        Additional keyword arguments passed to plot_coefficient_function().

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object.
    axes : array of matplotlib.axes.Axes
        Array of axes objects.

    Example
    -------
    >>> from fpls import fit_fpls, plot_comparison
    >>> import numpy as np
    >>> # Fit two datasets
    >>> X1 = np.random.randn(100, 50)
    >>> y1 = np.random.randn(100)
    >>> X2 = np.random.randn(100, 50)
    >>> y2 = np.random.randn(100)
    >>> coef1, _ = fit_fpls(X1, y1, m_max=5)
    >>> coef2, _ = fit_fpls(X2, y2, m_max=5)
    >>> s = np.linspace(0, 1, 50)
    >>> fig, axes = plot_comparison(
    ...     [s, s], [coef1[:, 3], coef2[:, 3]],
    ...     titles=["Dataset 1", "Dataset 2"]
    ... )
    >>> plt.show()
    """
    if style == "seaborn" and HAS_SEABORN:
        sns.set_style("ticks")
        sns.set_palette("husl")

    # Default colors
    if colors is None:
        colors = ["#2E86AB", "#E85D04", "#52B788", "#D62828", "#6A4C93"]

    n_plots = len(s_list)
    fig, axes = plt.subplots(1, n_plots, figsize=figsize, dpi=100)

    # Handle single plot case
    if n_plots == 1:
        axes = [axes]

    # Plot each coefficient function
    for i, (s, beta, title) in enumerate(zip(s_list, beta_list, titles)):
        color = colors[i % len(colors)]
        plot_coefficient_function(
            s,
            beta,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            color=color,
            fill_between=fill_between,
            style=style,
            ax=axes[i],
            **kwargs
        )

    # Add overall title if provided
    if suptitle:
        fig.suptitle(suptitle, fontsize=18, fontweight="bold", y=1.02)

    # Despine if seaborn is available
    if style == "seaborn" and HAS_SEABORN:
        sns.despine(offset=10, trim=True)

    plt.tight_layout()

    return fig, axes


def load_example_data(dataset="corn"):
    """
    Load example crop yield datasets from GitHub.

    Parameters
    ----------
    dataset : str, optional
        Dataset to load: 'corn' or 'soybeans'.

    Returns
    -------
    X : ndarray, shape (n_samples, n_features)
        Temperature exposure data (residualized).
    y : ndarray, shape (n_samples,)
        Log yield data (residualized).
    s : ndarray, shape (n_features,)
        Temperature grid (0-36°C).

    Example
    -------
    >>> from fpls import load_example_data, fit_fpls, plot_coefficient_function
    >>> X, y, s = load_example_data("corn")
    >>> coef, ds = fit_fpls(X, y, m_max=10)
    >>> plot_coefficient_function(s, coef[:, 4], xlabel="Temperature (°C)")
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("pandas is required to load example data. Install with: pip install pandas")

    base_url = "https://raw.githubusercontent.com/ababii/FPLS/main/data/"

    if dataset.lower() == "corn":
        url = base_url + "corn.csv"
        df = pd.read_csv(url)
        y = df["log_yield_corn_r"].values
        X = df.iloc[:, 1:].values
    elif dataset.lower() in ["soybean", "soybeans"]:
        url = base_url + "soybeans.csv"
        df = pd.read_csv(url)
        y = df["log_yield_soybeans_r"].values
        X = df.iloc[:, 1:].values
    else:
        raise ValueError(f"Unknown dataset: {dataset}. Choose 'corn' or 'soybeans'.")

    s = np.linspace(0, 36, X.shape[1])

    return X, y, s
