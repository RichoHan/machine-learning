#!/usr/bin/python
# -*- coding: big5 -*-
"""Session 1 Linear Regression"""
import numpy as np
import pandas as pd

from task_io import TaskIO


def score(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())


def get_processed_training(training_data, selection, window_width=5):
    features = training_data.loc[training_data['測項'].isin(selection)].reset_index(drop=True)
    max_width = 24 - window_width
    size = int(features.shape[0] / len(selection))
    x = np.zeros(shape=(size * max_width, window_width * len(selection)))
    y = np.zeros(shape=(size * max_width, 1))
    for index, row in features.iterrows():
        for i in range(0, max_width):
            samples = row.loc[str(i):str(i + window_width - 1)].apply(pd.to_numeric).values
            for pivot, sample in np.ndenumerate(samples):
                x[int(index / len(selection)) * max_width + i][(index % len(selection)) * window_width + pivot] = sample
            if row['測項'] == 'PM2.5':
                answer = row.loc[str(i + window_width):str(i + window_width)].apply(pd.to_numeric).values
                y[int(index / len(selection)) * max_width + i] = answer
    return x, y


def get_processed_testing(testing_data, selection, window_width=5):
    testing_features = testing_data.loc[testing_data[1].isin(selection)].reset_index(drop=True)
    size = int(testing_features.shape[0] / len(selection))
    x = np.zeros(shape=(size, window_width * len(selection)))
    for index, row in testing_features.iterrows():
        samples = row.iloc[(11 - window_width):11].apply(pd.to_numeric)
        for pivot, sample in np.ndenumerate(samples):
            x[int(index / len(selection))][(index % len(selection)) * window_width + pivot] = sample
    return x


def train_test_split(x, y, test_split=0, k=10):
    X_splits = np.split(x, k)
    X_train = None
    y_splits = np.split(y, k)
    y_train = None

    for i in range(0, k):
        if i != test_split:
            X_train = X_splits[i] if X_train is None else np.concatenate((X_train, X_splits[i]), axis=0)
            y_train = y_splits[i] if y_train is None else np.concatenate((y_train, y_splits[i]), axis=0)

    X_test = X_splits[test_split]
    y_test = y_splits[test_split]
    return X_train, X_test, y_train, y_test


def main():
    # ===== Suppressing warnings =====
    import warnings
    warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")

    # ===== Importing training and testing data =====
    task_io = TaskIO(
        train='./data/train.csv',
        test='./data/test_X.csv',
        result='./data/result.csv',
        validation_path='./data/validation.csv'
    )
    training_data = task_io.import_training_data()
    # testing_data = task_io.import_testing_data()

    # ===== Data Processing =====
    # selection = ['PM2.5', 'PM10', 'AMB_TEMP', 'O3']
    selection = ['PM2.5', 'PM10']
    # window_width = 5
    # x, y = get_processed_training(training_data, selection, window_width=window_width)

    # ===== Cross Validation =====
    window_sizes = list()
    training_scores = list()
    testing_scores = list()
    regularizations = list()

    from linear_model import LinearRegression
    for i in range(1, 10):
        window_width = i
        x, y = get_processed_training(training_data, selection, window_width=window_width)
        for split in range(10):
            X_train, X_test, y_train, y_test = train_test_split(x, y, test_split=split)
            regularization = 1000
            model = LinearRegression()
            model.fit(X_train, y_train, regularization)
            training_prediction = np.apply_along_axis(model.predict, 1, X_train)
            testing_prediction = np.apply_along_axis(model.predict, 1, X_test)

            window_sizes.append(i)
            training_scores.append(score(training_prediction, y_train))
            testing_scores.append(score(testing_prediction, y_test))
            regularizations.append(regularization)

    task_io.export_validation(pd.DataFrame({
        'window size': window_sizes,
        'regularization': regularizations,
        'training score': training_scores,
        'testing score': testing_scores
    }))

    # # ===== Fitting linear model =====
    # from linear_model import LinearRegression
    # model = LinearRegression()
    # regularization = 1000
    # model.fit(x, y, regularization)

    # # ===== Prediction =====
    # testing_x = get_processed_testing(testing_data, selection)
    # prediction = np.apply_along_axis(model.predict, 1, testing_x)

    # # ===== Exporting prediction result =====
    # ids = testing_data[testing_data[1] == 'PM10'].iloc[:, 0]
    # result = pd.concat(
    #     [
    #         ids.to_frame('id').reset_index(drop=True),
    #         pd.DataFrame.from_items([('value', prediction.flatten())]).reset_index(drop=True)
    #     ],
    #     axis=1,
    #     ignore_index=True
    # )
    # result.columns = ['id', 'value']

    # task_io.export_prediction(result)


if __name__ == "__main__":
    main()
