"""
Gaussian Discriminant Analysis (GDA) Classifier Module

Faithfully implements the generative Gaussian Discriminant Analysis model
described in Section IV.2 and Eq. (4) of the reference paper:
    p(y) = phi^y * (1 - phi)^{1-y}
    p(x|y=0) ~ N(mu_0, Sigma)
    p(x|y=1) ~ N(mu_1, Sigma)

Derives the closed-form posterior distribution and linear decision boundary:
    p(y=1|x) = sigmoid( theta^T x + theta_0 )
    where:
        theta = Sigma^{-1} * (mu_1 - mu_0)
        theta_0 = 0.5 * (mu_0^T Sigma^{-1} mu_0 - mu_1^T Sigma^{-1} mu_1) + ln(phi / (1 - phi))
"""

from typing import Optional, Tuple
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


class GaussianDiscriminantAnalysis(BaseEstimator, ClassifierMixin):
    """
    Gaussian Discriminant Analysis (Generative Linear Classifier).
    """

    def __init__(self, reg_covar: float = 1e-4):
        self.reg_covar = reg_covar
        self.phi_: Optional[float] = None
        self.mu0_: Optional[np.ndarray] = None
        self.mu1_: Optional[np.ndarray] = None
        self.sigma_: Optional[np.ndarray] = None
        self.sigma_inv_: Optional[np.ndarray] = None
        self.theta_: Optional[np.ndarray] = None
        self.theta_0_: Optional[float] = None
        self.coef_: Optional[np.ndarray] = None
        self.intercept_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        m, n = X.shape

        mask0 = (y == 0)
        mask1 = (y == 1)

        m0 = np.sum(mask0)
        m1 = np.sum(mask1)

        if m0 == 0 or m1 == 0:
            raise ValueError("GDA requires at least one example for each class.")

        # Class prior
        self.phi_ = float(m1 / m)

        # Class means
        self.mu0_ = np.mean(X[mask0], axis=0)
        self.mu1_ = np.mean(X[mask1], axis=0)

        # Shared covariance matrix Sigma:
        # Sigma = (1/m) * [ sum_{y=0} (x - mu0)(x - mu0)^T + sum_{y=1} (x - mu1)(x - mu1)^T ]
        diff0 = X[mask0] - self.mu0_
        diff1 = X[mask1] - self.mu1_

        sigma = (diff0.T @ diff0 + diff1.T @ diff1) / m
        # Add slight regularization to diagonal for numerical conditioning
        sigma += self.reg_covar * np.eye(n)
        self.sigma_ = sigma

        try:
            self.sigma_inv_ = np.linalg.inv(sigma)
        except np.linalg.LinAlgError:
            self.sigma_inv_ = np.linalg.pinv(sigma)

        # Linear decision boundary parameters matching Eq. 4 derivation:
        # theta = Sigma^{-1} * (mu_1 - mu_0)
        # theta_0 = 0.5 * (mu_0^T Sigma^{-1} mu_0 - mu_1^T Sigma^{-1} mu_1) + ln(phi / (1 - phi))
        self.theta_ = self.sigma_inv_ @ (self.mu1_ - self.mu0_)

        log_prior_ratio = np.log(self.phi_ / (1.0 - self.phi_))
        quad0 = float(self.mu0_.T @ self.sigma_inv_ @ self.mu0_)
        quad1 = float(self.mu1_.T @ self.sigma_inv_ @ self.mu1_)
        self.theta_0_ = float(0.5 * (quad0 - quad1) + log_prior_ratio)

        # For scikit-learn API compatibility:
        self.coef_ = np.array([self.theta_])
        self.intercept_ = np.array([self.theta_0_])

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.theta_ is None or self.theta_0_ is None:
            raise RuntimeError("Model is not fitted yet.")

        X = np.asarray(X, dtype=float)
        # Logit: z = X @ theta + theta_0
        z = X @ self.theta_ + self.theta_0_
        # Posterior probability: P(y=1|x) = sigmoid(z)
        p1 = 1.0 / (1.0 + np.exp(-np.clip(z, -250, 250)))
        p0 = 1.0 - p1
        return np.column_stack([p0, p1])

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        p1 = self.predict_proba(X)[:, 1]
        return (p1 >= threshold).astype(int)
