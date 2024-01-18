import pandas as pd
from abc import ABC, abstractmethod


class FeatureAnalysisCSVGenerator(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def create_csv(self, data: pd.DataFrame, file_path: str) -> None:
        pass


class CorrelationCSVGenerator(FeatureAnalysisCSVGenerator):
    def __init__(self):
        pass

    def create_csv(self, data: pd.DataFrame, file_path: str) -> None:
        try:
            correlation_matrix = data.corr()
            correlation_matrix.to_csv(file_path, index=True)
        except Exception as err:
            print("Unable to create correlation csv file, error: ", err)


class CovarianceCSVGenerator(FeatureAnalysisCSVGenerator):
    def __init__(self):
        pass

    def create_csv(self, data: pd.DataFrame, file_path: str) -> None:
        try:
            covariance_matrix = data.cov()
            covariance_matrix.to_csv(file_path, index=True)
        except Exception as err:
            print("Unable to create covariance csv file, error: ", err)
