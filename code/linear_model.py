"Linear model for assignment 2."
import numpy as np


class LogisticRegression():
    def __init__(self):
        self.feature_means = dict()
        self.feature_stds = dict()
        self.costs = list()
        self.thetas = list()
        self._lambda = 0

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def _cost(self, theta, X, y, epsilon=1e-10):
        # Calculate -lnL(w, b), add epsilon(1e-10) to avoid zero denominator.
        p1 = self._sigmoid(np.dot(X, theta))
        log_l = (-y) * np.log(p1) - (1 - y) * np.log(1 - p1 + epsilon)

        return log_l.sum() + self._lambda * (np.matrix(theta) * np.matrix(theta).T)

    def _gradient(self, theta, X, y):
        error = self._sigmoid(np.dot(X, theta)) - y
        grad = np.dot(error, X) / y.size

        return grad

    def _gradient_descent(self, theta, X, y, upper=20000, gamma=0.95, epsilon=1e-10):
        accu_grad = 0
        delta = 0
        accu_delta = 0
        t = 0

        last_cost = 0
        cost = self._cost(theta, X, y)
        while np.abs(cost - last_cost) > epsilon and t < upper:
            t += 1
            # Update gradient and accumulated gradient
            grad = self._gradient(theta, X, y)
            accu_grad = gamma * accu_grad + (1 - gamma) * grad * grad

            # Update delta and accumulated delta
            delta = -(np.sqrt((accu_delta + epsilon) / t) / np.sqrt((accu_grad + epsilon) / (t + 1))) * grad
            accu_delta = gamma * accu_delta + (1 - gamma) * delta * delta

            # Update theta
            theta += delta

            last_cost = cost
            cost = self._cost(theta, X, y)
            self.costs.append(cost)
            self.thetas.append(theta.copy())

        return theta

    def fit(self, X, y, _lambda=0):
        """Compute the fit function for the model.

        Parameters
        ----------
        X (ndarray): Training vector.

        y (ndarray): Target vector.

        Returns
        -------
        self : returns an instance of self.
        """
        # Apply z-score normalization on training set and store means & variance for testing data
        for feature in X.columns:
            self.feature_means[feature] = X[feature].mean()
            self.feature_stds[feature] = X[feature].std(ddof=0)
        X = X.apply(lambda col: (col - col.mean()) / col.std(ddof=0))

        n_params = X.shape[1]
        self._lambda = _lambda
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
        X = X.apply(lambda col: (col - self.feature_means[col.name]) / self.feature_stds[col.name])
        p = self._sigmoid(np.dot(X, self.theta))
        return [1 if x >= 0.5 else 0 for x in p]

    def get_recording(self, X, y, score):
        scores = list()
        for theta in self.thetas:
            p = self._sigmoid(np.dot(X, theta))
            scores.append(score(y, [1 if x >= 0.5 else 0 for x in p]))

        return self.costs, scores
