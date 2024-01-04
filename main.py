import pandas as pd
import numpy as np
import seaborn as sns
from pandas import DataFrame, Series
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from scipy.stats import zscore
from sklearn.model_selection import KFold
from stats_analysis_csv import CorrelationCSV, CovarianceCSV


def main():

    df = pd.read_csv('dataset/us_records_subset.csv')
    df = df.dropna()
    df_columns = list(df.columns)
    target_attributes_columns = df_columns[1: len(df_columns) - 2]

    df_subset_target_attributes = pd.DataFrame(
        data=df, columns=target_attributes_columns)

    print(df_subset_target_attributes.head())

    sns.set_theme(style="whitegrid")
    # sns.relplot(data=df_subset_target_attributes,
    #             x="total_cases", y="new_cases")

    corr_matrix = df_subset_target_attributes.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm')
    plt.show()

    fig, axs = plt.subplots(nrows=4, ncols=4, figsize=(12, 12))

    for i, var in enumerate(target_attributes_columns):
        row = i // 4
        col = i % 4
        sns.regplot(data=df_subset_target_attributes, x=var, y="icu_patients",
                    ax=axs[row, col], line_kws={'color': 'black'})

    plt.tight_layout()
    plt.show()

    # print(df_subset_target_attributes.head())
    scaler = StandardScaler()
    # print(np.mean(df_subset_target_attributes, 0))
    # print(np.std(df_subset_target_attributes, 0))

    standardized_target_attribute_df = pd.DataFrame(data=scaler.fit_transform(
        df_subset_target_attributes), columns=target_attributes_columns)
    # print(np.mean(standardized_target_attribute_df, 0))
    # print(np.std(standardized_target_attribute_df, 0))

    # correlation_csv = CorrelationCSV()
    # covariance_csv = CovarianceCSV()
    # covariance_csv.create_csv(standardized_target_attribute_df,
    #                           'output_statistics_csv/covariance.csv')
    # correlation_csv.create_csv(standardized_target_attribute_df,
    #                            'output_statistics_csv/correlation.csv')

    # Don't need to standardize the data meaning divide each col with it's standard deviation resulting in zero mean and 1 std
    # before_df = df_subset_target_attributes.corr()
    # after_df = standardized_target_attribute_df.corr()
    # print(before_df.round(3))
    # print(after_df.round(3))

    # print(np.array_equal(before_df.round(3), after_df.round(3)))
    # x = np.linspace(0, 2, 100)
    # plt.figure(figsize=(5, 5))
    # plt.plot(x, x, label="linear")
    # plt.plot(x, x**2, label="quadratic")
    # plt.title("Simple plot")
    # plt.legend()
    # plt.show()

    # print(std([85, 92, 64, 99, 56]))

    # print(df.corr())

    # print(df.count())
    # print(df['7day_avg_new_deaths'].head())

    # dates_df = pd.DataFrame(df, columns=['date'])
    # dates_df['date'] = pd.to_datetime(dates_df['date'], format='%m/%d/%Y')

    # attribute_col_names = list(df[df.columns[1: len(df.columns) - 2]])
    # attributes_df = pd.DataFrame(df, columns=attribute_col_names)

    # hospitalization_need_df = pd.DataFrame(
    #     df, columns=['hospitalization_need'])
    # icu_requirement_df = pd.DataFrame(
    #     df, columns=['icu_requirement'])

    # def plot_bar_chart(x, y, x_label, y_label, title):
    #     plt.bar(x, y)

    #     if x_label == 'Date':
    #         plt.gca().xaxis.set_major_formatter(
    #             plt.matplotlib.dates.DateFormatter('%m/%d/%Y'))

    #         plt.gcf().autofmt_xdate()

    #     plt.xlabel(x_label)
    #     plt.ylabel(y_label)
    #     plt.title(title)
    #     plt.show()

    # plot_bar_chart(dates_df['date'], attributes_df['new_cases'],
    #                'Date', 'New Cases', 'Daily New Cases Over Time')


main()
