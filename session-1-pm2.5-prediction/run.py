#!/usr/bin/python
# -*- coding: big5 -*-

"""Session 1 Linear Regression"""
import pandas as pd

if __name__ == "__main__":
    data = pd.read_csv('./data/train.csv', encoding="big5")
    features = data.groupby('測項')
    pm_25 = data[data['測項'] == 'PM2.5'].loc[:, '0':'23'].apply(pd.to_numeric)
    print(pm_25.describe())
    # y = pm_25.describe().loc['mean', :]

    # for (feature, group) in features:
    #     print(feature)
    #     values = group.loc[:, '0':'23']
    #     if feature == 'RAINFALL':
    #         values = values.replace('NR', '0')
    #     means = values.apply(pd.to_numeric).describe().loc['mean', :]
    #     print(means.describe())
