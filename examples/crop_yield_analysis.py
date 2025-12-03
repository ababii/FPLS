"""
Crop Yield Analysis Example
============================

This example demonstrates how to use FPLS to analyze the impact of temperature
exposure on crop yields using real agricultural data.

The data comes from the empirical application in:
    Babii, A., Carrasco, M., & Tsafack, I. (2025).
    "Functional Partial Least-Squares: Adaptive Estimation and Inference."
    Journal of the American Statistical Association.

The datasets contain:
- y: Log crop yield (residualized after controlling for other factors)
- X: Temperature exposure at each degree from 0°C to 36°C (residualized)
"""

import numpy as np
import matplotlib.pyplot as plt
from fpls import (
    load_example_data,
    fit_fpls,
    plot_coefficient_function,
    plot_comparison,
    compute_r2,
    compute_mse,
)


def main():
    # Load example datasets
    print("Loading corn and soybean data from GitHub...")
    X_corn, y_corn, s = load_example_data("corn")
    X_soy, y_soy, _ = load_example_data("soybeans")

    print(f"Corn data: {X_corn.shape[0]} observations, {X_corn.shape[1]} temperature bins")
    print(f"Soybean data: {X_soy.shape[0]} observations, {X_soy.shape[1]} temperature bins")
    print(f"Temperature range: {s[0]:.0f}°C to {s[-1]:.0f}°C\n")

    # Fit FPLS models
    print("Fitting FPLS models...")
    m_max = 10
    coef_corn, ds = fit_fpls(X_corn, y_corn, m_max=m_max, ds=1.0)
    coef_soy, _ = fit_fpls(X_soy, y_soy, m_max=m_max, ds=1.0)

    # Evaluate model performance for different numbers of components
    print("\nModel Performance (R²):")
    print("-" * 50)
    print(f"{'Components':<12} {'Corn':<12} {'Soybeans':<12}")
    print("-" * 50)

    for m in [1, 2, 3, 4, 5, 10]:
        # Corn predictions
        beta_corn = coef_corn[:, m]
        y_pred_corn = (X_corn @ beta_corn * ds)
        r2_corn = compute_r2(y_corn, y_pred_corn)

        # Soybean predictions
        beta_soy = coef_soy[:, m]
        y_pred_soy = (X_soy @ beta_soy * ds)
        r2_soy = compute_r2(y_soy, y_pred_soy)

        print(f"{m:<12} {r2_corn:<12.4f} {r2_soy:<12.4f}")

    # Use 4 components for visualization (good balance of fit and complexity)
    m_selected = 4
    beta_corn = coef_corn[:, m_selected]
    beta_soy = coef_soy[:, m_selected]

    print(f"\nUsing {m_selected} components for visualization\n")

    # Plot individual coefficient functions
    print("Plotting coefficient functions...")

    fig1, ax1 = plot_coefficient_function(
        s,
        beta_corn,
        title="Impact of Temperature on Corn Yield",
        xlabel="Temperature (°C)",
        ylabel="Log Yield (Bushels)",
        color="#2E86AB",
    )
    plt.savefig("corn_temperature_effect.png", dpi=300, bbox_inches="tight")
    print("Saved: corn_temperature_effect.png")

    fig2, ax2 = plot_coefficient_function(
        s,
        beta_soy,
        title="Impact of Temperature on Soybean Yield",
        xlabel="Temperature (°C)",
        ylabel="Log Yield (Bushels)",
        color="#E85D04",
    )
    plt.savefig("soybean_temperature_effect.png", dpi=300, bbox_inches="tight")
    print("Saved: soybean_temperature_effect.png")

    # Create side-by-side comparison
    fig3, axes = plot_comparison(
        s_list=[s, s],
        beta_list=[beta_corn, beta_soy],
        titles=[
            "Impact of Temperature on Corn Yield",
            "Impact of Temperature on Soybean Yield",
        ],
        xlabel="Temperature (°C)",
        ylabel="Log Yield (Bushels)",
        colors=["#2E86AB", "#E85D04"],
    )
    plt.savefig("crop_yield_comparison.png", dpi=300, bbox_inches="tight")
    print("Saved: crop_yield_comparison.png")

    # Interpretation
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    print(
        """
The coefficient functions show how temperature exposure affects crop yields:

1. Both crops show negative effects at extreme temperatures (>30°C)
2. Optimal temperature ranges differ between crops
3. The functional approach captures smooth nonlinear relationships
4. These patterns are consistent with agronomic knowledge about
   heat stress effects on crop production

The FPLS method automatically learns these temperature-yield relationships
from the data without imposing parametric functional forms.
"""
    )

    plt.show()


if __name__ == "__main__":
    main()
