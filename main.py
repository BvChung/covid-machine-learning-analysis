import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from feature_analysis_csv_generator import CorrelationCSVGenerator, CovarianceCSVGenerator
from regression_metrics import TrainingSetRegressionMetrics, TestingSetRegressionMetrics, DisplayRegressionMetrics
from sklearn.model_selection import train_test_split


def main():
    us_covid_records_df = pd.read_csv('dataset/us_records_subset.csv')
    us_covid_records_df = us_covid_records_df.dropna()
    us_covid_records_df_columns = ['date', 'total_cases', 'new_cases', 'total_deaths', 'new_deaths', 'total_cases_per_million', 'total_deaths_per_million', 'icu_patients', 'hosp_patients', 'weekly_hosp_admissions',
                                   'daily_case_change_rate', 'daily_death_change_rate', 'hospitalization_rate', 'icu_rate', 'case_fatality_rate', '7day_avg_new_cases', '7day_avg_new_deaths', 'hospitalization_need', 'icu_requirement']

    numerical_attribute_columns = us_covid_records_df_columns[1: len(
        us_covid_records_df_columns) - 2]
    df_numerical_attributes_subset = pd.DataFrame(
        data=us_covid_records_df, columns=numerical_attribute_columns)

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
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, shuffle=True, random_state=42)

    display_metrics_table = DisplayRegressionMetrics()
    training_set_metrics_calculator = TrainingSetRegressionMetrics(
        X_train=X_train, y_train=y_train, k_folds=10)
    testing_set_metrics_calculator = TestingSetRegressionMetrics(
        X_test=X_test, y_test=y_test)

    lin_reg_model = make_pipeline(
        StandardScaler(), LinearRegression()).fit(X_train, y_train)

    lin_reg_training_set_metrics = training_set_metrics_calculator.get_regression_metrics(
        lin_reg_model)
    display_metrics_table.display_table(
        metrics=lin_reg_training_set_metrics, title="Least Squares Linear Regression Training Metrics")

    linear_coef = lin_reg_model.named_steps['linearregression'].coef_
    y_intercept = lin_reg_model.named_steps['linearregression'].intercept_
    print(f"Linear Regression Coefficients: {linear_coef}")
    print(f"Linear Regression Y Intercept: {y_intercept}")

    lin_reg_testing_set_metrics = testing_set_metrics_calculator.get_regression_metrics(
        lin_reg_model)
    display_metrics_table.display_table(
        lin_reg_testing_set_metrics, title="Least Squares Linear Regression Final Metrics")

    lasso_model = make_pipeline(StandardScaler(), LassoCV(cv=10))
    lasso_model.fit(X_train, y_train)

    lasso_training_set_metrics = training_set_metrics_calculator.get_regression_metrics(
        lasso_model)
    display_metrics_table.display_table(
        metrics=lasso_training_set_metrics, title="Lasso Linear Regression Training Metrics")

    lasso_testing_set_metrics = testing_set_metrics_calculator.get_regression_metrics(
        lasso_model)
    display_metrics_table.display_table(
        lasso_testing_set_metrics, title="Lasso Linear Regression Final Metrics")

    ridge_model = make_pipeline(StandardScaler(), RidgeCV(cv=10))
    ridge_model.fit(X_train, y_train)

    ridge_regression_metrics = training_set_metrics_calculator.get_regression_metrics(
        ridge_model)
    display_metrics_table.display_table(
        metrics=ridge_regression_metrics, title="Ridge Linear Regression Training Metrics")

    ridge_testing_set_metrics = testing_set_metrics_calculator.get_regression_metrics(
        ridge_model)
    display_metrics_table.display_table(
        ridge_testing_set_metrics, title="Ridge Linear Regression Final Metrics")


if __name__ == "__main__":
    main()
