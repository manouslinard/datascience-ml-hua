import numpy as np

class LinearRegression:

    def __init__(self) -> None:
        self.w = None
        self.b = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Calculates weights and beta."""
        # Check for X and y np array and if X and y are compatible.
        if not isinstance(X, np.ndarray):
            raise ValueError("X is not numpy array.")
        if not isinstance(y, np.ndarray):
            raise ValueError("y is not numpy array.")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y are not compatible.")

        ones = np.full((X.shape[0], 1), 1.0)
        X = np.append(X, ones, axis=1)

        Xt = np.transpose(X)

        theta = np.dot(np.dot(np.linalg.inv(np.dot(Xt, X)), Xt), y)
        self.w = theta[:-1]  # all elements except from last.
        self.b = theta[-1]  # last element.

    def predict(self, X: np.ndarray):
        """Returns the predictions of the model."""
        if not isinstance(X, np.ndarray):
            raise ValueError("X is not numpy array.")
        if self.w is None or self.b is None:
            raise ValueError("Model is not trained (use fit first).")
        if X.shape[1] != self.w.shape[0]:   # X is Nxp and w is px1.
            raise ValueError("Model not trained for specified X matrix.")
        return np.dot(X, self.w) + self.b

    def evaluate(self, X: np.ndarray, y: np.ndarray):
        """
        Returns the predictions and the MSE.
        """
        if not isinstance(y, np.ndarray):
            raise ValueError("y is not numpy array.")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y are not compatible.")
        N = X.shape[0]

        yhat = self.predict(X)

        MSE = (1 / N) * np.dot(np.transpose(yhat - y), (yhat - y))
        return yhat, MSE
