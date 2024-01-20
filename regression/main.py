import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import PredictionErrorDisplay
from feature_analysis_csv_generator import CorrelationCSVGenerator, CovarianceCSVGenerator
from regression_metrics import TrainingSetRegressionMetrics, TestingSetRegressionMetrics, DisplayRegressionMetrics
from regression_coefficients import PlotRegressionCoefficients
from sklearn.model_selection import train_test_split
from display_plots import display_regression_plot, display_heatmap


def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    file_path = os.path.join(parent_dir, 'dataset', 'us_records_subset.csv')

    sns.set_theme(style="whitegrid")

    us_covid_records_df = pd.read_csv(file_path)
    us_covid_records_df = us_covid_records_df.dropna()
    us_covid_records_columns = ['date', 'total_cases', 'new_cases', 'total_deaths', 'new_deaths', 'total_cases_per_million', 'total_deaths_per_million', 'icu_patients', 'hosp_patients', 'weekly_hosp_admissions',
                                'daily_case_change_rate', 'daily_death_change_rate', 'hospitalization_rate', 'icu_rate', 'case_fatality_rate', '7day_avg_new_cases', '7day_avg_new_deaths', 'hospitalization_need', 'icu_requirement']

    numerical_attribute_columns = us_covid_records_columns[1: len(
        us_covid_records_columns) - 2]
    numerical_attributes_subset_df = pd.DataFrame(
        data=us_covid_records_df, columns=numerical_attribute_columns)

    corr_matrix = numerical_attributes_subset_df.corr()
    display_heatmap(corr_matrix, (10, 8))

    display_regression_plot(numerical_attributes_subset_df,
                            "icu_patients", numerical_attribute_columns, 4, 4, (12, 12))

    target_features = ['new_cases', 'new_deaths', 'hosp_patients', 'weekly_hosp_admissions',
                       'daily_case_change_rate', 'daily_death_change_rate', '7day_avg_new_cases', '7day_avg_new_deaths']

    X = numerical_attributes_subset_df[target_features]
    y = numerical_attributes_subset_df['icu_patients']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, shuffle=True, random_state=42)

    display_metrics_table = DisplayRegressionMetrics()
    plot_regression_coefficients = PlotRegressionCoefficients()

    training_set_metrics_calculator = TrainingSetRegressionMetrics(
        X_train=X_train, y_train=y_train, k_folds=10)
    testing_set_metrics_calculator = TestingSetRegressionMetrics(
        X_test=X_test, y_test=y_test)

    # Linear Regression Model
    lin_reg_model = make_pipeline(
        StandardScaler(), LinearRegression()).fit(X_train, y_train)

    lin_reg_training_set_metrics = training_set_metrics_calculator.get_regression_metrics(
        lin_reg_model)
    display_metrics_table.display_table(
        metrics=lin_reg_training_set_metrics, title="Least Squares Linear Regression Training Metrics")

    lin_reg_testing_set_metrics = testing_set_metrics_calculator.get_regression_metrics(
        lin_reg_model)
    display_metrics_table.display_table(
        metrics=lin_reg_testing_set_metrics, title="Least Squares Linear Regression Final Metrics")

    linear_coefs = lin_reg_model.named_steps['linearregression'].coef_
    y_intercept = lin_reg_model.named_steps['linearregression'].intercept_
    feature_names = lin_reg_model[:-1].get_feature_names_out()

    plot_regression_coefficients.plot(
        coefs=linear_coefs, y_intercept=y_intercept, feature_names=feature_names, model_name="Linear")

    # Lasso Regression Model
    lasso_model = make_pipeline(
        StandardScaler(), LassoCV(cv=10)).fit(X_train, y_train)
    lasso_y_pred = lasso_model.predict(X_test)

    lasso_training_set_metrics = training_set_metrics_calculator.get_regression_metrics(
        lasso_model)
    display_metrics_table.display_table(
        metrics=lasso_training_set_metrics, title="Lasso Linear Regression Training Metrics")

    lasso_testing_set_metrics = testing_set_metrics_calculator.get_regression_metrics(
        lasso_model)
    display_metrics_table.display_table(
        lasso_testing_set_metrics, title="Lasso Linear Regression Final Metrics")

    print(
        f'Best performing alpha: {lasso_model.named_steps["lassocv"].alpha_}')

    _, ax = plt.subplots(1, 1)
    display = PredictionErrorDisplay.from_predictions(
        y_test, lasso_y_pred, kind="actual_vs_predicted", ax=ax, scatter_kwargs={'alpha': 0.5})
    ax.set_title("Lasso model, optimum regularization")
    plt.show()

    lasso_coefs = lasso_model.named_steps['lassocv'].coef_
    y_intercept = lasso_model.named_steps['lassocv'].intercept_
    feature_names = lasso_model[:-1].get_feature_names_out()

    plot_regression_coefficients.plot(
        coefs=lasso_coefs, y_intercept=y_intercept, feature_names=feature_names, model_name="Lasso")

    # Ridge Regression Model
    ridge_model = make_pipeline(
        StandardScaler(), RidgeCV(cv=10)).fit(X_train, y_train)

    ridge_regression_metrics = training_set_metrics_calculator.get_regression_metrics(
        ridge_model)
    display_metrics_table.display_table(
        metrics=ridge_regression_metrics, title="Ridge Linear Regression Training Metrics")

    ridge_testing_set_metrics = testing_set_metrics_calculator.get_regression_metrics(
        ridge_model)
    display_metrics_table.display_table(
        ridge_testing_set_metrics, title="Ridge Linear Regression Final Metrics")

    ridge_coefs = ridge_model.named_steps['ridgecv'].coef_
    y_intercept = ridge_model.named_steps['ridgecv'].intercept_
    feature_names = ridge_model[:-1].get_feature_names_out()

    plot_regression_coefficients.plot(
        coefs=ridge_coefs, y_intercept=y_intercept, feature_names=feature_names, model_name="Ridge")


if __name__ == "__main__":
    main()
