"Linear model for assignment 1."
from sklearn.linear_model import LinearRegression


class LinearRegression(LinearRegression):
    def __init__(self):
        super().__init__()

    def gradient_descent(eta, x, y, end_point=0.0001, max_iter=10000):
        """Gradient descent is used to calculate the best fitting function for the model.

        Args:
            eta (float): Learning rate.
            x (ndarray): The training input.
            y (ndarray): The training output.
            end_point (float): Convergence criteria.
            max_iter (int): Maximum number of iteration.

        Returns:
            tuple: A pair of intercept(=theta0) and slope(=theta1).
        """
        pass
