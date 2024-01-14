import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, make_scorer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate


def main():
    us_covid_records_df = pd.read_csv('dataset/us_records_subset.csv')
    us_covid_records_df.dropna(inplace=True)

    hospitalization_need_labels = us_covid_records_df['hospitalization_need']
    icu_requirement_labels = us_covid_records_df['icu_requirement']

    us_covid_records_df.drop(
        columns=['date', 'hospitalization_need', 'icu_requirement'], inplace=True)

    X = us_covid_records_df
    y = hospitalization_need_labels
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, shuffle=True, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.fit_transform(X_test)

    svm_rbf_kernel_model = SVC(kernel='rbf', random_state=42)
    svm_rbf_kernel_model.fit(X_train_scaled, y_train)
    acc = svm_rbf_kernel_model.score(X_test_scaled, y_test)
    print(f'Mean Accuracy: {acc:.3f}')

    stratkf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    scoring = {
        'accuracy': make_scorer(accuracy_score),
        'recall': make_scorer(recall_score, average='weighted'),
        'precision': make_scorer(precision_score, average='weighted', zero_division=1)
    }

    scores = cross_validate(
        svm_rbf_kernel_model, X_train_scaled, y_train, cv=stratkf, scoring=scoring)

    table_data = [['Accuracy', round(scores["test_accuracy"].mean(), 3)],
                  ['Precision', round(scores["test_precision"].mean(), 3)],
                  ['Recall', round(scores["test_recall"].mean(), 3)]]

    fig, ax = plt.subplots(1, 1)
    table = plt.table(cellText=table_data, colLabels=[
        'Metric', 'Mean Value'], loc='center')
    table.set_fontsize(11)
    table.scale(1.1, 1.4)
    ax.set_title("SVM RBF Kernel Cross Validation Scores")
    ax.axis('off')
    plt.show()

    # Test prediction
    point = us_covid_records_df.iloc[[10]].values
    print(svm_rbf_kernel_model.predict(point))


if __name__ == "__main__":
    main()
