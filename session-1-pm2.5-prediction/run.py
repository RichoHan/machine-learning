#!/usr/bin/python
# -*- coding: big5 -*-
"""Session 1 Linear Regression"""
import numpy as np
import pandas as pd

from task_io import TaskIO


def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())


if __name__ == "__main__":
    # ===== Suppressing warnings =====
    import warnings
    warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")

    # ===== Importing training and testing data =====
    task_io = TaskIO(
        train='./data/train.csv',
        test='./data/test_X.csv',
        result='./data/result.csv'
    )
    training_data = task_io.import_training_data()
    testing_data = task_io.import_testing_data()

    # ===== Data manipulation =====
    features = training_data.groupby('測項')
    for feature, feature_data in features:
        if feature == 'PM2.5':
            pm_25 = feature_data.loc[:, '0':'23'].apply(pd.to_numeric)
            window_width = pm_25.shape[1] - 9
            x = np.zeros(shape=(pm_25.shape[0] * window_width, 9))
            y = np.zeros(shape=(pm_25.shape[0] * window_width, 1))
            for i in range(0, 15):
                samples = pm_25.loc[:, str(i):str(i + 8)].values
                answers = pm_25.loc[:, str(i + 9)].values
                for index, sample in np.ndenumerate(samples):
                    x[samples.shape[0] * i + index[0]][index[1]] = sample
                for index, answer in np.ndenumerate(answers):
                    y[samples.shape[0] * i + index[0]] = answer

    # ===== Fitting linear model =====
    from linear_model import LinearRegression
    model = LinearRegression()

    model.fit(x, y)

    # ===== Prediction =====
    testing_features = testing_data.groupby(1)
    for feature, feature_data in testing_features:
        if feature == 'PM2.5':
            pm_25 = feature_data.iloc[:, 2:11].apply(pd.to_numeric).values
    prediction = np.apply_along_axis(model.predict, 1, pm_25)

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
