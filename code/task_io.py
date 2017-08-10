# -*- coding:utf-8 -*-
from abc import ABCMeta, abstractmethod
import pickle


class TaskIO(metaclass=ABCMeta):
    def __init__(self, train, test, result, validation):
        self.train = train
        self.test = test
        self.result = result
        self.validation = validation

    def import_training_data(self):
        with open(self.train, 'rb') as fo:
            training_data = pickle.load(fo, encoding='bytes')
        return training_data

    def import_testing_data(self):
        with open(self.test, 'rb') as fo:
            testing_data = pickle.load(fo, encoding='bytes')
        return testing_data

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
