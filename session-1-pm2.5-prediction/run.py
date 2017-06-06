#!/usr/bin/python
# -*- coding: big5 -*-
"""Session 1 Linear Regression"""
import numpy as np
import pandas as pd

from task_io import TaskIO
from linear_model import LinearRegression


class Session1TaskIO(TaskIO):
    def get_processed_training(self, training_data, selection, window_width=4):
        features = training_data.loc[training_data['測項'].isin(selection)].reset_index(drop=True)
        pm_25 = training_data.loc[training_data['測項'] == 'PM2.5'].reset_index(drop=True)
        max_width = 24 - window_width
        size = int((features.shape[0]) / len(selection))
        x = np.zeros(shape=(size * max_width, window_width * len(selection)))
        y = np.zeros(shape=(size * max_width, 1))
        for index, row in features.iterrows():
            for i in range(max_width):
                samples = row.loc[str(i):str(i + window_width - 1)].apply(pd.to_numeric).values
                for pivot, sample in np.ndenumerate(samples):
                    x[int(index / len(selection)) * max_width + i][(index % len(selection)) * window_width + pivot] = sample
        for index, row in pm_25.iterrows():
            for i in range(max_width):
                answer = row.loc[str(i + window_width):str(i + window_width)].apply(pd.to_numeric).values
                y[int(index / len(selection)) * max_width + i] = answer
        return x, y

    def get_processed_testing(self, testing_data, selection, window_width=5):
        testing_features = testing_data.loc[testing_data[1].isin(selection)].reset_index(drop=True)
        size = int(testing_features.shape[0] / len(selection))
        x = np.zeros(shape=(size, window_width * len(selection)))
        for index, row in testing_features.iterrows():
            samples = row.iloc[(11 - window_width):11].apply(pd.to_numeric)
            for pivot, sample in np.ndenumerate(samples):
                x[int(index / len(selection))][(index % len(selection)) * window_width + pivot] = sample
        return x


def score(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())


def main():
    # ===== Suppressing warnings =====
    import warnings
    warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")

    # ===== Importing training and testing data =====
    task_io = Session1TaskIO(
        train='./data/train.csv',
        test='./data/test_X.csv',
        result='./data/result.csv',
        validation_path='./data/validation.csv'
    )
    training_data = task_io.import_training_data()
    testing_data = task_io.import_testing_data()

    # ===== Data Processing =====
    # selection = ['PM2.5', 'PM10', 'O3', 'NO2']
    selection = ['PM2.5', 'PM10']
    window_width = 4
    x, y = task_io.get_processed_training(training_data, selection, window_width=window_width)

    # ===== Fitting linear model =====
    model = LinearRegression()
    regularization = 10000
    model.fit(x, y, regularization)

    # ===== Prediction =====
    testing_x = task_io.get_processed_testing(testing_data, selection, window_width)
    prediction = np.apply_along_axis(model.predict, 1, testing_x)

    # ===== Exporting prediction result =====
    ids = testing_data[testing_data[1] == 'PM10'].iloc[:, 0]
    result = pd.concat(
        [
            ids.to_frame('id').reset_index(drop=True),
            pd.DataFrame.from_items([('value', prediction.flatten())]).reset_index(drop=True)
        ],
        axis=1,
        ignore_index=True
    )
    result.columns = ['id', 'value']

    task_io.export_prediction(result)


if __name__ == "__main__":
    main()
