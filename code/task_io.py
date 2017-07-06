from abc import ABCMeta, abstractmethod

import pandas as pd


class TaskIO(metaclass=ABCMeta):
    def __init__(self, train, test, result, validation):
        self.train = train
        self.test = test
        self.result = result
        self.validation = validation

    def import_training_data(self, header=None, names=None):
        return pd.read_csv(self.train, header=header, names=names, encoding="big5")

    def import_testing_data(self, header=None, names=None):
        return pd.read_csv(self.test, header=header, names=names, encoding="big5")

    def export_prediction(self, data):
        data.to_csv(self.result, index=False)

    def export_validation(self, data):
        data.to_csv(self.validation, index=False)

    @abstractmethod
    def get_processed_training(self):
        pass

    @abstractmethod
    def get_processed_testing(self):
        pass

    @abstractmethod
    def score(self):
        pass
