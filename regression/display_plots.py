import matplotlib.pyplot as plt
from pandas import DataFrame
import seaborn as sns


def display_heatmap(df: DataFrame, fig_size: tuple[int, int]):
    plt.figure(figsize=fig_size)
    sns.heatmap(df, annot=True, fmt=".2f", cmap='coolwarm')
    plt.show()


def display_regression_plot(df: DataFrame, predicted_var: str, feature_columns: list, n_rows: int, n_cols: int, fig_size: tuple[int, int]):
    fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=fig_size)

    for i, feature in enumerate(feature_columns):
        row = i // n_rows
        col = i % n_cols
        sns.regplot(data=df, x=feature, y=predicted_var,
                    ax=axs[row, col], line_kws={'color': 'black'})

    plt.tight_layout()
    plt.show()
