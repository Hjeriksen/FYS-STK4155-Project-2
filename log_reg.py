import autograd.numpy as np
from autograd import grad


class LogisticRegression:

    def __init__(self, learning_rate=0.01, epochs=10, lambda_=0):
        self.learning_rate = learning_rate
        self.lambda_ = lambda_
        self.epochs = epochs
        self.beta = None
        self.intercept = None

    def sigmoid(self, z):
        """Compute the sigmoid function."""
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        """
        Train the logistic regression model using gradient descent.

        Parameters:
        X : np.array, shape (m, n)
            Input data, where m is the number of samples and n is the number of features.
        y : np.array, shape (m,)
            Labels (0 or 1) for the input data.
        """
        # Initialize beta and intercept
        m, n = X.shape
        self.beta = np.random.rand(n)
        self.intercept = np.random.rand()

        # Gradient descent
        for i in range(self.epochs):

            for j in range(len(X)):

                cost = lambda beta_, intercept_: np.power( self.sigmoid( X[j].dot(beta_) + intercept_ ) - y[j], 2) + self.lambda_ * (np.sum(np.power(beta_, 2)) + np.power(intercept_, 2))

                der_beta = grad(cost, 0)(self.beta, self.intercept)
                der_intercept = grad(cost, 1)(self.beta, self.intercept)

                self.beta -= self.learning_rate * der_beta
                self.intercept -= self.learning_rate * der_intercept

    def predict_proba(self, X):
        """Return the probability estimates for each sample in X."""
        linear_model = np.dot(X, self.beta) + self.intercept
        return self.sigmoid(linear_model)

    def predict(self, X):
        """
        Predict binary labels (0 or 1) for each sample in X.

        Parameters:
        X : np.array, shape (m, n)
            Input data.

        Returns:
        np.array, shape (m,)
            Predicted binary labels (0 or 1).
        """
        y_pred_proba = self.predict_proba(X)
        return np.where(y_pred_proba >= 0.5, 1, 0).reshape(-1, 1)
