from numpy import ndarray
import pandas as pd
import matplotlib.pyplot as plt


class RegressionCoefficients:
    def __init__(self):
        pass

    def _get_regression_equation(self, coefs: ndarray, y_intercept: float) -> str:

        equation = f"y = {y_intercept:.3f}"
        for i in range(len(coefs)):
            operator = '-' if coefs[i] < 0 else '+'
            value = abs(coefs[i]) if coefs[i] < 0 else coefs[i]
            equation += f' {operator} {value:.3f} x_{i + 1}'

        return equation

    def plot_coefficients(self, coefs: ndarray, y_intercept: float, feature_names: ndarray, model_name: str) -> None:
        coefs_df = pd.DataFrame(
            coefs, columns=['Coefficient Importance'], index=feature_names)

        regression_equation = self._get_regression_equation(
            coefs, y_intercept)

        coefs_df.plot.barh(figsize=(9, 7))
        plt.title(f"{model_name} Regression Model Coefficients")
        plt.axvline(x=0, color=".5")
        plt.xlabel("Coefficient values")
        plt.subplots_adjust(left=0.3, bottom=0.2)
        plt.figtext(
            0.5, 0.02, f"Equation: {regression_equation}", ha="center", fontsize=12, wrap=True,
            bbox={"boxstyle": "round, pad=1", "facecolor": "white", "alpha": 0.5, "edgecolor": "black"})

        plt.show()
