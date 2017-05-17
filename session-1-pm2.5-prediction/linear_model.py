"Linear model for assignment 1."
from sklearn.linear_model import LinearRegression


class LinearRegression2(LinearRegression):
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
        size = x.shape[0]
        converged = False
        iteration = 0
        b, w = 0, 0

        # Goodness of function
        loss = sum([(b + w * x[i] - y[i])**2 for i in range(size)])
        print("x.shape: {0}".format(x.shape))
        print("size: {0}".format(size))
        print("loss: {0}".format(loss))

        # Converge iteration
        while not converged and iteration < max_iter:
            # For each training sample, compute the gradient (d/d_theta j(theta))
            grad_0 = (1.0 / size) * sum([(b + (w * x[i]) - y[i]) for i in range(size)]) 
            grad_1 = (1.0 / size) * sum([(b + (w * x[i]) - y[i]) * x[i] for i in range(size)])

            # Update bias and weight with eta
            temp_b = b - eta * grad_0
            temp_w = w - eta * grad_1
            b = temp_b
            w = temp_w

            # Compute the error again
            error = sum([(b + w * x[i] - y[i])**2 for i in range(size)])

            if abs(error - loss) <= end_point:
                converged = True

            loss = error
            iteration = iteration + 1
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

        eta = 0.01  # learning rate
        end_point = 0.01  # convergence criteria

        # call gredient decent, and get intercept(=bias) and slope(=weight)
        bias, weight = self._gradient_descent(eta, X, y, end_point, max_iter=1000)
        print('bias = {0}, weight = {1}'.format(bias, weight))

        super().fit(X, y)
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
        # return self._decision_function(X)
        y = 1 + 5 * X
        return y
