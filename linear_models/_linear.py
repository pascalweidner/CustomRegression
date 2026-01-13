import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted


class LinearRegression(BaseEstimator, RegressorMixin):
    def __init__(self, fit_intercept=True):
        self.fit_intercept = fit_intercept

    def fit(self, X, y):
        X, y = check_X_y(X, y)

        if self.fit_intercept:
            X = np.hstack([np.ones((X.shape[0], 1)), X])

        coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)

        if self.fit_intercept:
            self.intercept_ = coeffs[0]
            self.coef_ = coeffs[1:]
        else:
            self.intercept_ = 0
            self.coef_ = coeffs

        return self

    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X)

        return X @ self.coef_ + self.intercept_
