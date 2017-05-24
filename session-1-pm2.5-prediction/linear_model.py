"Linear model for assignment 1."
import numpy as np


class LinearRegression():
    def __init__(self):
        super().__init__()

    def _gradient_descent(self, eta, x, y, end_point=0.0001, max_iter=10000):
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
        print(size, width)
        converged = False
        iteration = 0
        b, w = 1, np.random.rand(1, width)

        # Goodness of function
        def diff(x, y):
            return y - (b + np.dot(w, x))
        loss = sum([diff(x[i], y[i])**2 for i in range(size)])
        print(loss)

        # Converge iteration
        while not converged and iteration < max_iter:
            # For each training sample, compute the gradient (d/d_theta j(theta))
            grad_0 = sum(2 * [diff(x[i], y[i]) for i in range(size)])
            grad_1 = sum(2 * [diff(x[i], y[i]) * (-x[i]) for i in range(size)])

            # Update bias and weight with eta
            temp_b = b - eta * grad_0
            temp_w = w - eta * grad_1
            b = temp_b
            w = temp_w

            # Compute the error again
            error = sum([diff(x[i], y[i])**2 for i in range(size)])

            if abs(error - loss) <= end_point:
                print("Converged at iteration {0}".format(iteration))
                converged = True

            loss = error
            iteration = iteration + 1
            print(loss)
        return b, w

    def fit(self, X, y):
        """Compute the fit function for the model.

        Parameters
        ----------
        X (ndarray): Training data.

        y (ndarray): Target value.

        Returns
        -------
        self : returns an instance of self.
        """

        eta = 0.000000005  # learning rate
        end_point = 0.01  # convergence criteria

        # Call gredient decent, and get intercept(=bias) and slope(=weight)
        self.bias, self.weight = self._gradient_descent(eta, X, y, end_point, max_iter=10000)
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
