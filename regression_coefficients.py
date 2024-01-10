from numpy import ndarray
import pandas as pd
import matplotlib.pyplot as plt


class RegressionCoefficients:
    def __init__(self):
        pass

    def print_coefficients(self, coefs: ndarray, y_intercept: float, feature_names: ndarray, model_name: str) -> None:
        print(f"{model_name} Regression Coefficients: {coefs}")
        print(f"{model_name} Regression Y Intercept: {y_intercept}")
        print(f"{model_name} Regression Feature Names: {feature_names}")

    def plot_coefficients(self, coefs: ndarray, feature_names: ndarray, model_name: str) -> None:
        lasso_coefs_df = pd.DataFrame(
            coefs, columns=['Coefficient Importance'], index=feature_names)

        lasso_coefs_df.plot.barh(figsize=(9, 7))
        plt.title(f"{model_name} Regression Model Coefficients")
        plt.axvline(x=0, color=".5")
        plt.xlabel("Coefficient values")
        plt.subplots_adjust(left=0.3)
        plt.show()
