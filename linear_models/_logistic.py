import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted


def logistic_loss(w, X, y):
    z = X@w
    return np.sum(np.logaddexp(0, z) - y * z)


def logistic_gradient(w, X, y):
    z = X @ w
    p = expit(z)
    return X.T @ (p - y)


class LogisticRegression(BaseEstimator, ClassifierMixin):
    def __init__(self, fit_intercept=True, max_iter=100):
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter

    def fit(self, X, y):
        X, y = check_X_y(X, y)

        if self.fit_intercept:
            X = np.hstack([np.ones((X.shape[0], 1)), X])

        n_features = X.shape[1]

        result = minimize(
            fun=logistic_loss,
            x0=np.zeros(n_features),
            args=(X, y),
            jac=logistic_gradient,
            method='L-BFGS-B',
            options={"maxiter": self.max_iter}
        )

        coeffs = result.x
        if self.fit_intercept:
            self.intercept_ = coeffs[0]
            self.coef_ = coeffs[1:]
        else:
            self.intercept_ = 0
            self.coef_ = coeffs

        return self

    def predict_proba(self, X):
        check_is_fitted(self)
        X = check_array(X)

        z = X @ self.coef_ + self.intercept_
        return expit(z)

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)