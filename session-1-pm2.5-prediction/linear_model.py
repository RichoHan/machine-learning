"Linear model for assignment 1."
import numpy as np


class LinearRegression():
    def __init__(self):
        super().__init__()

    def _gradient_descent(self, eta, x, y, end_point=0.0001, max_iter=10000, regularization=0):
        """Gradient descent is used to calculate the best fitting function: y = b + wx

        Parameters
        ----------
        eta (float): Learning rate.

        x (ndarray): Training data.

        y (ndarray): Target value.

        end_point (float): Convergence criteria.

        max_iter (int): Maximum number of iteration.

        Returns
        -------
        tuple: A pair of intercept(=bias) and slope(=weight).
        """
        # Initialization
        size, width = x.shape
        converged = False
        iteration = 1
        b, w = 1, np.random.rand(1, width)

        # Goodness of function
        def diff(x, y):
            return y - (b + np.dot(w, x))

        def loss_f(x, y, size):
            return sum([diff(x[i], y[i])**2 for i in range(size)])

        loss = loss_f(x, y, size) + regularization * np.sum(np.apply_along_axis(lambda x: x**2, 0, w))

        # Converge iteration
        while not converged and iteration < max_iter:
            # Stochastic gradient descent
            for i in range(size):
                grad_0 = 2 * diff(x[i], y[i]) * (-1)
                grad_1 = 2 * diff(x[i], y[i]) * (-x[i])
                temp_b = b - eta * grad_0
                temp_w = w - eta * grad_1
                b = temp_b
                w = temp_w

            # Compute the error again
            error = loss_f(x, y, size) + regularization * np.sum(np.apply_along_axis(lambda x: x**2, 0, w))

            # if error > loss:
            if abs(error - loss) > 1e+8:
                print("Diverged at iteration {0}".format(iteration))
                # return b, w, converged

            if abs(error - loss) <= end_point:
                print("Converged at iteration {0}".format(iteration))
                converged = True

            loss = error
            iteration = iteration + 1
        return b, w, True

    def fit(self, X, y, regularization=0):
        """Compute the fit function for the model.

        Parameters
        ----------
        X (ndarray): Training data.

        y (ndarray): Target value.

        Returns
        -------
        self : returns an instance of self.
        """
        eta = 0.000001  # learning rate
        end_point = 0.01  # convergence criteria
        converged = False

        # Call gredient decent, and get intercept(=bias) and slope(=weight)
        while not converged:
            print("Setting eta to {0}".format(eta))
            self.bias, self.weight, converged = self._gradient_descent(eta, X, y, end_point, max_iter=100, regularization=regularization)
            eta = eta / 2
        # self.bias, self.weight, converged = self._gradient_descent(eta, X, y, end_point, max_iter=100, regularization=regularization)
        print('bias = {0}, weight = {1}'.format(self.bias, self.weight))

        return self

    def predict(self, X):
        """Predict using the linear model

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = (n_samples, n_features)
            Samples.

        Returns
        -------
        C : array, shape = (n_samples,)
            Returns predicted values.
        """
        # Calculate predictions using fitted model
        y = self.bias + np.dot(self.weight, X)
        return y
