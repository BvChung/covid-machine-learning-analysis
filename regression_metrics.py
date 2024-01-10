from numpy import ndarray
from pandas import DataFrame
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, KFold
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from abc import ABC, abstractmethod


class RegressionMetrics(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def _get_r2_score(self, model: Pipeline) -> ndarray:
        pass

    @abstractmethod
    def _get_mse_score(self, model: Pipeline) -> ndarray:
        pass

    @abstractmethod
    def _get_mas_score(self, model: Pipeline) -> ndarray:
        pass

    @abstractmethod
    def get_regression_metrics(self, model: Pipeline) -> dict[str, float]:
        pass


class TrainingSetRegressionMetrics(RegressionMetrics):
    def __init__(self, X_train: DataFrame, y_train: DataFrame,  k_folds: int = 10):
        self.k_folds = KFold(n_splits=k_folds, shuffle=True, random_state=42)
        self.X_train = X_train
        self.y_train = y_train

    def _get_r2_score(self, model: Pipeline) -> ndarray:
        return cross_val_score(model, self.X_train, self.y_train, scoring='r2', cv=self.k_folds)

    def _get_mse_score(self, model: Pipeline) -> ndarray:
        return cross_val_score(model, self.X_train, self.y_train, scoring='neg_mean_squared_error', cv=self.k_folds)

    def _get_mas_score(self, model: Pipeline) -> ndarray:
        return cross_val_score(model, self.X_train, self.y_train, scoring='neg_mean_absolute_error', cv=self.k_folds)

    def get_regression_metrics(self, model: Pipeline) -> dict[str, float]:
        r2_scores = self._get_r2_score(model)
        mse_scores = self._get_mse_score(model)
        mas_scores = self._get_mas_score(model)

        return {
            "mean_training_set_r2_score": f'{r2_scores.mean():.3f} +/- {r2_scores.std():.3f}',
            "mean_training_set_mse_score": f'{abs(mse_scores.mean()):.3f} +/- {abs(mse_scores.std()):.3f}',
            "mean_training_set_mas_score": f'{abs(mas_scores.mean()):.3f} +/- {abs(mas_scores.std()):.3f}'
        }


class TestingSetRegressionMetrics(RegressionMetrics):
    def __init__(self, X_test: DataFrame, y_test: DataFrame):
        self.X_test = X_test
        self.y_test = y_test

    def _get_r2_score(self, y_pred: ndarray) -> float:
        return r2_score(self.y_test, y_pred)

    def _get_mse_score(self, y_pred: ndarray) -> float:
        return mean_squared_error(self.y_test, y_pred)

    def _get_mas_score(self, y_pred: ndarray) -> float:
        return mean_absolute_error(self.y_test, y_pred)

    def get_regression_metrics(self, model: Pipeline) -> dict[str, float]:
        y_pred = model.predict(self.X_test)
        r2_score = self._get_r2_score(y_pred)
        mse_score = self._get_mse_score(y_pred)
        mas_score = self._get_mas_score(y_pred)

        return {
            "test_set_r2_score": f'{r2_score:.3f}',
            "test_set_mse_score": f'{mse_score:.3f}',
            "test_set_mas_score": f'{mas_score:.3f}'
        }


class DisplayRegressionMetrics:
    def __init__(self):
        pass

    def display_table(self, metrics: dict[str, float], title: str) -> None:
        fig, ax = plt.subplots(1, 1)

        table_data = []
        for key, value in metrics.items():
            table_data.append([key, value])

        table = plt.table(cellText=table_data, loc='center')

        table.set_fontsize(12)
        table.scale(1.2, 1.5)

        ax.set_title(title)

        ax.axis('off')

        plt.show()
