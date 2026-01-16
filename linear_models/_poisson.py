import numpy as np
from scipy.optimize import minimize
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted


def poisson_loss(w, X, y):
    X_w = X @ w
    return -np.sum(X_w * y - np.exp(X_w))


def poisson_gradient(w, X, y):
    return (np.exp(X @ w) - y) @ X


def poisson_deviance(y_true, y_pred):
    y_pred = np.maximum(y_pred, 1e-10)

    term1 = np.where(y_true > 0, y_true * np.log(y_true / y_pred), 0)
    term2 = y_true - y_pred
    return 2 * np.sum(term1 - term2)


class PoissonRegression(BaseEstimator, RegressorMixin):
    def __init__(self, fit_intercept=True, max_iter=100):
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter

    def fit(self, X, y):
        X, y = check_X_y(X, y)

        if self.fit_intercept:
            X = np.hstack([np.ones((X.shape[0], 1)), X])

        n_features = X.shape[1]

        result = minimize(
            fun=poisson_loss,
            x0=np.zeros(n_features),
            args=(X, y),
            jac=poisson_gradient,
            method='L-BFGS-B',
            options={"maxiter": self.max_iter}
        )

        coeffs = result.x
        if self.fit_intercept:
            self.intercept_ = coeffs[0]
            self.coef_ = coeffs[1:]
        else:
            self.intercept_ = 0.0
            self.coef_ = coeffs

    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X)

        return np.exp(X @ self.coef_ + self.intercept_)

    def score(self, X, y, sample_weight=None):
        check_is_fitted()
        X, y = check_X_y(X, y)

        y_pred = self.predict(X)

        dev_model = poisson_deviance(y, y_pred)

        y_null = np.full_like(y, fill_value=np.mean(y))
        dev_null = poisson_deviance(y, y_null)

        return 1 - (dev_model / dev_null)
