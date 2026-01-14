import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted


class LassoRegression(BaseEstimator, RegressorMixin):
    def __init__(self, lmbda=0.0, fit_intercept=True, tol=1e-4, max_iter=1000):
        self.fit_intercept = fit_intercept
        self.tol = tol
        self.max_iter = max_iter
        self.lmbda = lmbda

    def fit(self, X, y):
        X, y = check_X_y(X, y)

        if self.fit_intercept:
            X = np.hstack([np.ones((X.shape[0], 1)), X])

        n_features = X.shape[1]
        n_samples = X.shape[0]

        coeffs = np.zeros(n_features)
        y_hat = np.zeros(n_samples)

        for iteration in range(self.max_iter):
            coeffs_old = coeffs.copy()

            for j in range(n_features):
                y_hat_no_j = y_hat - coeffs[j] * X[:, j]

                rho_j = np.dot(X[:, j], y - y_hat_no_j)

                z_j = np.sum(X[:, j]**2)
                if z_j < 1e-15:
                    coeffs[j] = 0.0
                    continue

                if self.fit_intercept and j == 0:
                    coeffs[j] = rho_j / z_j
                else:
                    if rho_j > self.lmbda:
                        coeffs[j] = (rho_j - self.lmbda) / z_j
                    elif rho_j < -self.lmbda:
                        coeffs[j] = (rho_j + self.lmbda) / z_j
                    else:
                        coeffs[j] = 0.0

                y_hat = y_hat_no_j + coeffs[j] * X[:, j]

            diff = np.max(np.abs(coeffs - coeffs_old))

            if diff < self.tol:
                break

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
