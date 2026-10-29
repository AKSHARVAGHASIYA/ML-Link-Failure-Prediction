"""
Machine Learning Models Package
Exposes:
- Logistic Regression (Newton's Method & Scikit-learn wrapper)
- Gaussian Discriminant Analysis (GDA)
- Support Vector Machine with Linear Kernel
"""

from .logistic_regression import NewtonLogisticRegression, build_logistic_regression
from .gda import GaussianDiscriminantAnalysis
from .linear_svm import build_linear_svm

__all__ = [
    "NewtonLogisticRegression",
    "build_logistic_regression",
    "GaussianDiscriminantAnalysis",
    "build_linear_svm",
]
