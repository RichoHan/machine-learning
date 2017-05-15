#!/usr/bin/python
# -*- coding: big5 -*-

"""Session 1 Linear Regression"""
import pandas as pd
import matplotlib.dates as mdates
import numpy as np
from datetime import datetime

if __name__ == "__main__":
    data = pd.read_csv('./data/train.csv', encoding="big5")
    pm_25_date = data[data['����'] == 'PM2.5'].loc[:, '���']
    dates = [datetime.strptime(date, '%Y/%m/%d') for date in pm_25_date.values]

    pm_25 = data[data['����'] == 'PM2.5'].loc[:, '0':'23'].apply(pd.to_numeric)
    daily_pm_25 = pm_25.mean(axis=1)
    print(pm_25.describe())
