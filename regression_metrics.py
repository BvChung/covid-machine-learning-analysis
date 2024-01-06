from numpy import ndarray
from pandas import DataFrame
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, KFold
import matplotlib.pyplot as plt


class RegressionMetrics:
    def __init__(self, X: DataFrame, y: DataFrame,  k_folds: int = 10):
        self.k_folds = KFold(n_splits=k_folds, shuffle=True, random_state=42)
        self.X = X
        self.y = y

    def _get_r2_score(self, pipe: Pipeline) -> ndarray:
        return cross_val_score(pipe, self.X, self.y, scoring='r2', cv=self.k_folds)

    def _get_mse_score(self, pipe: Pipeline) -> ndarray:
        return cross_val_score(pipe, self.X, self.y, scoring='neg_mean_squared_error', cv=self.k_folds)

    def _get_mas_score(self, pipe: Pipeline) -> ndarray:
        return cross_val_score(pipe, self.X, self.y, scoring='neg_mean_absolute_error', cv=self.k_folds)

    def get_regression_metrics(self, pipe: Pipeline) -> dict[str, float]:
        r2_scores = self._get_r2_score(pipe)
        mse_scores = self._get_mse_score(pipe)
        mas_scores = self._get_mas_score(pipe)

        return {
            "mean_r2_score": r2_scores.mean(),
            "mean_mse_score": abs(mse_scores).mean(),
            "mean_mas_score": abs(mas_scores).mean()
        }


class DisplayMetrics:
    def __init__(self):
        self.metrics_name_map = {
            "mean_r2_score": "Mean R2 Score",
            "mean_mse_score": "Mean MSE Score",
            "mean_mas_score": "Mean MAS Score"
        }

    def display_metrics(self, metrics: dict[str, float], title: str) -> None:
        fig, ax = plt.subplots(1, 1)

        table_data = []
        for key, value in metrics.items():
            table_data.append([key, round(value, 3)])

        table = plt.table(cellText=table_data, loc='center')

        table.set_fontsize(12)
        table.scale(1, 1.5)

        ax.set_title(title)

        ax.axis('off')

        plt.show()
