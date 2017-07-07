"Linear model for assignment 2."
import numpy as np

from sklearn import linear_model


class LogisticRegression(linear_model.LogisticRegression):
    def __init__(self):
        super().__init__()

    def _sigmoid(self, X):
        return 1.0 / (1.0 + np.exp(-X))

    def _cost(self, theta, X, y):
        pass

    def _gradient(self, theta, X, y):
        pass

    def fit(self, X, y):
        """Compute the fit function for the model.

        Parameters
        ----------
        X (ndarray): Training vector.

        y (ndarray): Target vector.

        Returns
        -------
        self : returns an instance of self.
        """
        self._cost(0, X, y)
        super().fit(X, y)

    def predict(self, X):
        """Predict using the linear model

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Samples.

        Returns
        -------
        C : array, shape = (n_samples,)
            Class label per sample.
        """
        return super().predict(X)
