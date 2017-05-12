#!/usr/bin/python
# -*- coding: big5 -*-

"""Session 1 Linear Regression"""
import pandas as pd
import matplotlib.dates as mdates
import numpy as np
from datetime import datetime

if __name__ == "__main__":
    data = pd.read_csv('./data/train.csv', encoding="big5")
    pm_25_date = data[data['測項'] == 'PM2.5'].loc[:, '日期']
    dates = [datetime.strptime(date, '%Y/%m/%d') for date in pm_25_date.values]

    pm_25 = data[data['測項'] == 'PM2.5'].loc[:, '0':'23'].apply(pd.to_numeric)

    features = data.groupby('測項')
    for (feature, group) in features:
        print(feature)
        values = group.loc[:, '0':'23']
        if feature == 'RAINFALL':
            values = values.replace('NR', '0')
        means = values.apply(pd.to_numeric).describe().loc['mean', :]
        print(means.describe())
