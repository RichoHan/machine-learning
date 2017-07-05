"Linear model for assignment 2."
from sklearn import linear_model


class LogisticRegression(linear_model.LogisticRegression):
    def __init__(self):
        super().__init__()

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
