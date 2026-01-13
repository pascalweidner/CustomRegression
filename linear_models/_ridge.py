import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted


class RidgeRegression(BaseEstimator, RegressorMixin):
    def __init__(self, alpha=0.0, fit_intercept=True):
        self.fit_intercept = fit_intercept
        self.alpha = alpha

    def fit(self, X, y):
        X, y = check_X_y(X, y)

        if self.fit_intercept:
            X = np.hstack([np.ones((X.shape[0], 1)), X])

        n_features = X.shape[1]

        A = self.alpha * np.eye(n_features)
        if self.fit_intercept:
            A[0, 0] = 0

        coeffs = np.linalg.solve(X.T @ X + A, X.T @ y)

        if self.fit_intercept:
            self.intercept_ = coeffs[0]
            self.coef_ = coeffs[1:]
        else:
            self.intercept_ = 0.0
            self.coef_ = coeffs

        return self

    def predict(self, X):
        check_is_fitted(self)
        check_array(X)

        return X @ self.coef_ + self.intercept_
