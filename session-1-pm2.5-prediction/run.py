#!/usr/bin/python
# -*- coding: big5 -*-

"""Session 1 Linear Regression"""
import numpy as np
import pandas as pd

from task_io import TaskIO


class Session1TaskIO(TaskIO):
    def export_prediction(self, data):
        print('\n===== Exporting prediction result... =====')
        super().export_prediction(data)


def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())


if __name__ == "__main__":
    # ===== Suppressing warnings =====
    import warnings
    warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")

    # ===== Importing training and testing data =====
    task_io = Session1TaskIO(
        train='./data/train.csv',
        test='./data/test_X.csv',
        result='./data/result.csv'
    )
    training_data = task_io.import_training_data()
    testing_data = task_io.import_testing_data()

    # ===== Data manipulation =====
    features = training_data.groupby('測項')
    pm_25 = training_data[training_data['測項'] == 'PM2.5'].loc[:, '0':'23'].apply(pd.to_numeric)

    x = pd.Series()
    y = pd.Series()

    feature = training_data[training_data['測項'] == 'PM10'].loc[:, '0':'23'].apply(pd.to_numeric)
    outcome = pm_25
    for hour in feature:
        x = x.append(feature[hour].map(lambda elem: elem), ignore_index=True)
        y = y.append(outcome[hour].map(lambda elem: elem), ignore_index=True)

    # ===== Fitting linear model =====
    from linear_model import LinearRegression2
    model = LinearRegression2()

    model.fit(x.values, y.values)

    xfit = np.linspace(0, feature.max().max(), 1000)
    yfit = model.predict(xfit[:, np.newaxis])

    # ===== Prediction =====
    pm_10 = testing_data[testing_data[1] == 'PM10'].iloc[:, 2:11].apply(pd.to_numeric)
    # pm_25_answer = testing_data[testing_data[1] == 'PM2.5'].iloc[:, 11].apply(pd.to_numeric)
    prediction_from_pm_10 = pm_10.apply(
        lambda row: row[10],
        axis=1
    )
    prediction_from_pm_10 = model.predict(prediction_from_pm_10).astype('int')

    # print("\n===== prediction_from_pm_10 =====")
    # print(prediction_from_pm_10.values)
    # print("\n===== pm_2.5_answer =====")
    # print(pm_25_answer.values)
    # print("\n===== RMSE =====")
    # print(rmse(prediction_from_pm_10.values, pm_25_answer))

    # ===== Exporting prediction result =====
    ids = testing_data[testing_data[1] == 'PM10'].iloc[:, 0]
    result = pd.concat(
        [
            ids.to_frame('id').reset_index(drop=True),
            pd.DataFrame.from_items([('value', prediction_from_pm_10)]).reset_index(drop=True)
        ],
        axis=1,
        ignore_index=True
    )
    result.columns = ['id', 'value']

    task_io.export_prediction(result)
