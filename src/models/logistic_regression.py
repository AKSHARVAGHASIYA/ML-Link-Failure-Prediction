"""
Logistic Regression Classifier Module

Faithfully implements the baseline Logistic Regression model described in
Section IV.1 of the paper, trained via Newton's Method (Eq. 3):
    J(theta) = - (1/m) * sum( y*log(h(x)) + (1-y)*log(1-h(x)) )
    theta <- theta - H^{-1} * grad J(theta)
"""

from typing import Optional, Tuple
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, ClassifierMixin


class NewtonLogisticRegression(BaseEstimator, ClassifierMixin):
    """
    Logistic Regression classifier trained using Newton's Method (Newton-Raphson),
    as specifically referenced in Section IV.1 of the paper.
    """

    def __init__(self, max_iter: int = 50, tol: float = 1e-6, reg: float = 1e-4):
        self.max_iter = max_iter
        self.tol = tol
        self.reg = reg
        self.theta_: Optional[np.ndarray] = None
        self.loss_history_: list = []

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(z, -250, 250)))

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        m, n = X.shape

        # Add intercept column (bias term x0 = 1)
        X_design = np.c_[np.ones((m, 1)), X]
        theta = np.zeros(n + 1)

        self.loss_history_ = []

        for iteration in range(self.max_iter):
            # Hypothesis: h_theta(x) = sigmoid(X * theta)
            h = self._sigmoid(X_design @ theta)
            eps = 1e-15
            loss = -np.mean(y * np.log(h + eps) + (1.0 - y) * np.log(1.0 - h + eps))
            self.loss_history_.append(loss)

            # Gradient: grad = (1/m) * X^T (h - y)
            grad = (1.0 / m) * (X_design.T @ (h - y)) + self.reg * theta

            # Hessian: H = (1/m) * X^T * D * X, where D = diag(h * (1 - h))
            D = h * (1.0 - h)
            # Efficient vectorized computation of X^T * D * X:
            H = (1.0 / m) * (X_design.T * D) @ X_design + self.reg * np.eye(n + 1)

            # Newton step: delta = H^{-1} * grad
            try:
                delta = np.linalg.solve(H, grad)
            except np.linalg.LinAlgError:
                delta = np.linalg.pinv(H) @ grad

            theta -= delta

            if np.linalg.norm(delta) < self.tol:
                break

        self.theta_ = theta
        self.intercept_ = np.array([theta[0]])
        self.coef_ = np.array([theta[1:]])
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.theta_ is None:
            raise RuntimeError("Model is not fitted yet.")
        X = np.asarray(X, dtype=float)
        m = X.shape[0]
        X_design = np.c_[np.ones((m, 1)), X]
        p1 = self._sigmoid(X_design @ self.theta_)
        p0 = 1.0 - p1
        return np.column_stack([p0, p1])

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        prob = self.predict_proba(X)[:, 1]
        return (prob >= threshold).astype(int)


def build_logistic_regression(solver: str = "newton-cg", C: float = 1.0) -> LogisticRegression:
    """
    Returns a production scikit-learn LogisticRegression instance configured with
    Newton's method solver (newton-cg) matching paper specifications.
    """
    return LogisticRegression(
        solver=solver,
        C=C,
        max_iter=1000,
        random_state=42,
    )
