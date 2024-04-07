import matplotlib.pyplot as plt


def display_table(table_data: list[list[str]], column_labels: list[str], title: str) -> None:
    fig, ax = plt.subplots(1, 1)
    table = plt.table(cellText=table_data,
                      colLabels=column_labels, loc='center')

    table.set_fontsize(11)
    table.scale(1.1, 1.4)
    ax.set_title(title)
    ax.axis('off')
    plt.show()
