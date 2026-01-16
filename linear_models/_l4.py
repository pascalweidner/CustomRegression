import numpy as np
from scipy.optimize import minimize
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted


def l4_loss_no_intercept(w, X, y, lmbda):
    X_w = X @ w - y
    return 1/2 * X_w.T @ X_w + lmbda * np.sum(w[1:]**4)


def l4_gradient_no_intercept(w, X, y, lmbda):
    l4_grad = 4 * lmbda * (w**3)
    l4_grad[0] = 0.0
    return X.T @ (X @ w - y) + l4_grad


def l4_loss(w, X, y, lmbda):
    X_w = X @ w - y
    return 1/2 * X_w.T @ X_w + lmbda * np.sum(w**4)


def l4_gradient(w, X, y, lmbda):
    l4_grad = 4 * lmbda * (w**3)
    return X.T @ (X @ w - y) + l4_grad


class L4Regression(BaseEstimator, RegressorMixin):
    def __init__(self, fit_intercept=True, lmbda=1.0, max_iter=100):
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.lmbda = lmbda

    def fit(self, X, y):
        X, y = check_X_y(X, y)

        if self.fit_intercept:
            X = np.hstack([np.ones((X.shape[0], 1)), X])

        n_features = X.shape[1]

        active_loss = l4_loss_no_intercept if self.fit_intercept else l4_loss
        active_grad = l4_gradient_no_intercept if self.fit_intercept else l4_gradient

        result = minimize(
            fun=active_loss,
            x0=np.zeros(n_features),
            args=(X, y, self.lmbda),
            jac=active_grad,
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

        return X @ self.coef_ + self.intercept_
