from abc import ABCMeta, abstractmethod

import pandas as pd
import numpy as np


class TaskIO(metaclass=ABCMeta):
    def __init__(self, train, test, result, validation_path):
        self.train = train
        self.test = test
        self.result = result
        self.validation_path = validation_path

    def import_training_data(self):
        return pd.read_csv(self.train, encoding="big5")

    def import_testing_data(self):
        return pd.read_csv(self.test, encoding="big5", header=None)

    def export_prediction(self, data):
        data.to_csv(self.result, index=False)

    def export_validation(self, data):
        data.to_csv(self.validation_path, index=False)

    def train_test_split(self, x, y, test_split=0, k=10):
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

    @abstractmethod
    def get_processed_training(self):
        pass

    @abstractmethod
    def get_processed_testing(self):
        pass
