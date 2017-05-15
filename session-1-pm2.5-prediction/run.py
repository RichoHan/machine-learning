#!/usr/bin/python
# -*- coding: big5 -*-

"""Session 1 Linear Regression"""
import pandas as pd

if __name__ == "__main__":
    data = pd.read_csv('./data/train.csv', encoding="big5")
    features = data.groupby('測項')
    pm_25 = data[data['測項'] == 'PM2.5'].loc[:, '0':'23'].apply(pd.to_numeric)
    x = pd.Series()
    y = pd.Series()
    for hour in pm_25:
        x = x.append(pm_25[hour].map(lambda yy: hour), ignore_index=True)
        y = y.append(pm_25[hour].map(lambda yy: yy), ignore_index=True)
    print(type(x))