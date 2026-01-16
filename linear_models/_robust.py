import numpy as np
from scipy.optimize import linprog
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted


class RobustRegression(BaseEstimator, RegressorMixin):
    def __init__(self, fit_intercept=True):
        self.fit_intercept = fit_intercept

    def fit(self, X, y):
        X, y = check_X_y(X, y)

        if self.fit_intercept:
            X = np.hstack([np.ones((X.shape[0], 1)), X])

        n_samples, n_features = X.shape

        c = np.concatenate([np.zeros(n_features), np.ones(n_samples)])

        A_upper = np.hstack([-X, -np.eye(n_samples)])
        b_upper = -y

        A_lower = np.hstack([X, -np.eye(n_samples)])
        b_lower = y

        A = np.vstack([A_upper, A_lower])
        b = np.concatenate([b_upper, b_lower])

        bounds = [(None, None)] * n_features + [(0, None)] * n_samples

        res = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method="highs")

        if res.success:
            if self.fit_intercept:
                self.intercept_ = res.x[0]
                self.coef_ = res.x[1:n_features]
            else:
                self.intercept_ = 0.0
                self.coef_ = res.x[:n_features]
        else:
            raise ValueError("Optimization failed!")

        return self

    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X)

        return X @ self.coef_ + self.intercept_
