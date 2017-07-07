"Linear model for assignment 2."
import numpy as np

from sklearn import linear_model


class LogisticRegression(linear_model.LogisticRegression):
    def __init__(self):
        super().__init__()

    def _sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-z))

    def _cost(self, theta, X, y):
        # Prepare to do matrix multiplication
        theta = np.matrix(theta)
        X = np.matrix(X)
        y = np.matrix(y)

        # Calculate -lnL(w, b)
        p1 = np.multiply(-y, np.log(self._sigmoid(X * theta.T)))
        p2 = np.multiply((1 - y), np.log(1 - self._sigmoid(X * theta.T)))
        return np.sum(p1 - p2) / (len(X))

    def _gradient(self, theta, X, y):
        # Prepare to do matrix multiplication
        theta = np.matrix(theta)
        X = np.matrix(X)
        y = np.matrix(y)
        n_params = theta.shape[1]

        error = self._sigmoid(X * theta.T) - y
        vfunc = np.vectorize(lambda i: np.sum(np.multiply(error, X[:, i])) / len(X))
        indice = np.array(range(n_params))
        grad = vfunc(indice)

        return grad

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
        n_params = X.shape[1]
        theta = np.zeros(n_params)
        print(self._cost(theta, X, y))
        print(self._gradient(theta, X, y))
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
