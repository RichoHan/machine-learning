#!/usr/bin/python
# -*- coding: big5 -*-

"""Session 1 Linear Regression"""
import numpy as np
import pandas as pd

from task_io import TaskIO


class Session1TaskIO(TaskIO):
    def export_prediction(self, data):
        print(data.describe())
        data.to_csv(self.result, index=False)


if __name__ == "__main__":
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

    # x = pd.Series()
    # y = pd.Series()

    # feature = data[data['測項'] == 'PM10'].loc[:, '0':'23'].apply(pd.to_numeric)
    # outcome = pm_25
    # for hour in feature:
    #     x = x.append(feature[hour].map(lambda elem: elem), ignore_index=True)

    # for hour in outcome:
    #     y = y.append(outcome[hour].map(lambda elem: elem), ignore_index=True)

    # from sklearn.linear_model import LinearRegression
    # model = LinearRegression(fit_intercept=True)

    # model.fit(x[:, np.newaxis], y)

    # xfit = np.linspace(0, feature.max().max(), 1000)
    # yfit = model.predict(xfit[:, np.newaxis])
    # print(xfit)

    result = pd.DataFrame(columns=['id', 'value'])
    result = result.append(
        pd.DataFrame([['id_0', 0]], columns=['id', 'value']),
        ignore_index=True
    )
    task_io.export_prediction(result)
