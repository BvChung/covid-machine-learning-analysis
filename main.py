import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from covid_analysis_machine_learning.feature_analysis_csv_generator import CorrelationCSV, CovarianceCSV
from regression_metrics import RegressionMetrics, DisplayMetrics


def main():
    covid_df = pd.read_csv('dataset/us_records_subset.csv')
    covid_df = covid_df.dropna()
    covid_df_columns = ['date', 'total_cases', 'new_cases', 'total_deaths', 'new_deaths', 'total_cases_per_million', 'total_deaths_per_million', 'icu_patients', 'hosp_patients', 'weekly_hosp_admissions',
                        'daily_case_change_rate', 'daily_death_change_rate', 'hospitalization_rate', 'icu_rate', 'case_fatality_rate', '7day_avg_new_cases', '7day_avg_new_deaths', 'hospitalization_need', 'icu_requirement']

    numerical_attribute_columns = covid_df_columns[1: len(
        covid_df_columns) - 2]
    df_numerical_attributes_subset = pd.DataFrame(
        data=covid_df, columns=numerical_attribute_columns)

    sns.set_theme(style="whitegrid")

    corr_matrix = df_numerical_attributes_subset.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm')
    plt.show()

    fig, axs = plt.subplots(nrows=4, ncols=4, figsize=(12, 12))

    for i, var in enumerate(numerical_attribute_columns):
        row = i // 4
        col = i % 4
        sns.regplot(data=df_numerical_attributes_subset, x=var, y="icu_patients",
                    ax=axs[row, col], line_kws={'color': 'black'})

    plt.tight_layout()
    plt.show()

    target_features = ['new_cases', 'new_deaths', 'hosp_patients', 'weekly_hosp_admissions',
                       'daily_case_change_rate', 'daily_death_change_rate', '7day_avg_new_cases', '7day_avg_new_deaths']

    X = df_numerical_attributes_subset[target_features]
    y = df_numerical_attributes_subset['icu_patients']
    display_metrics = DisplayMetrics()
    regression_metrics = RegressionMetrics(X=X, y=y, k_folds=10)

    least_squares_pipe = make_pipeline(StandardScaler(), LinearRegression())
    least_squares_regression_metrics = regression_metrics.get_regression_metrics(
        least_squares_pipe)
    display_metrics.display_metrics(
        metrics=least_squares_regression_metrics, title="Least Squares Linear Regression Metrics")

    lasso_pipe = make_pipeline(StandardScaler(), LassoCV(cv=10))
    lasso_regression_metrics = regression_metrics.get_regression_metrics(
        lasso_pipe)
    display_metrics.display_metrics(
        metrics=lasso_regression_metrics, title="Lasso Linear Regression Metrics")

    ridge_pipe = make_pipeline(StandardScaler(), RidgeCV(cv=10))
    ridge_regression_metrics = regression_metrics.get_regression_metrics(
        ridge_pipe)
    display_metrics.display_metrics(
        metrics=ridge_regression_metrics, title="Ridge Linear Regression Metrics")


main()
