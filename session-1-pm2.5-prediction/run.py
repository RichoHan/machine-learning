#!/usr/bin/python
# -*- coding: big5 -*-

"""Session 1 Linear Regression"""
import numpy as np
import pandas as pd

from task_io import TaskIO


class Session1TaskIO(TaskIO):
    def export_prediction(self, data):
        print('===== Exporting prediction result... =====')
        data.to_csv(self.result, index=False)


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")
    task_io = Session1TaskIO(
        train='./data/train.csv',
        test='./data/test_X.csv',
        result='./data/result.csv'
    )
    training_data = task_io.import_training_data()
    testing_data = task_io.import_testing_data()

    features = training_data.groupby('測項')
    pm_25 = training_data[training_data['測項'] == 'PM2.5'].loc[:, '0':'23'].apply(pd.to_numeric)
    # print(pm_25.describe())
    # x = pd.Series()
    # y = pd.Series()
    # for hour in pm_25:
    #     x = x.append(pm_25[hour].map(lambda yy: hour), ignore_index=True)
    #     y = y.append(pm_25[hour].map(lambda yy: yy), ignore_index=True)

    x = pd.Series()
    y = pd.Series()

    feature = training_data[training_data['測項'] == 'PM10'].loc[:, '0':'23'].apply(pd.to_numeric)
    outcome = pm_25
    for hour in feature:
        x = x.append(feature[hour].map(lambda elem: elem), ignore_index=True)

    for hour in outcome:
        y = y.append(outcome[hour].map(lambda elem: elem), ignore_index=True)

    from sklearn.linear_model import LinearRegression
    model = LinearRegression(fit_intercept=True)

    model.fit(x[:, np.newaxis], y)

    xfit = np.linspace(0, feature.max().max(), 1000)
    yfit = model.predict(xfit[:, np.newaxis])

    pm_10 = testing_data[testing_data[1] == 'PM2.5'].iloc[:, 2:11].apply(pd.to_numeric)
    prediction_from_pm_10 = pm_10.apply(
        lambda row: row[10],
        axis=1
    )
    prediction_from_pm_10 = model.predict(prediction_from_pm_10[:, np.newaxis])
    prediction_from_pm_10 = prediction_from_pm_10.astype('int')

    ids = testing_data[testing_data[1] == 'PM2.5'].iloc[:, 0]
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
