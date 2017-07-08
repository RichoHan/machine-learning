"Linear model for assignment 2."
import numpy as np

from sklearn import linear_model


class LogisticRegression(linear_model.LogisticRegression):
    def __init__(self):
        super().__init__()

    def _sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-z))

    def _cost(self, theta, X, y):
        p1 = self._sigmoid(np.dot(X, theta))

        # Calculate -lnL(w, b), add 1e-10 to make sure denominator is no zero.
        log_l = (-y) * np.log(p1) - (1 - y) * np.log(1 - p1 + 1e-10)

        return log_l.mean()

    def _gradient(self, theta, X, y):
        error = self._sigmoid(np.dot(X, theta)) - y
        grad = np.dot(error, X) / y.size

        return grad

    def _gradient_descent(self, theta, X, y):
        import scipy.optimize as opt
        result = opt.minimize(
            fun=self._cost,
            x0=theta,
            args=(X, y),
            method='BFGS',
            jac=self._gradient
        )
        return result['x']

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
        self.theta = np.zeros(n_params, dtype=np.float128)
        self.theta = self._gradient_descent(self.theta, X, y)

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
        p = self._sigmoid(np.dot(X, self.theta))
        return [1 if x >= 0.5 else 0 for x in p]
