"""
Support Vector Machine with Linear Kernel Module

Implements the Linear Support Vector Machine classifier described in Section IV.3
and Eq. (5) of the reference paper:
    min_{w, b, xi} (1/2) * ||w||^2 + C * sum(xi_i)
    subject to: y_i * (w^T x_i + b) >= 1 - xi_i
"""

from sklearn.svm import SVC


def build_linear_svm(C: float = 1.0, random_state: int = 42) -> SVC:
    """
    Returns an SVM classifier with a linear kernel and Platt scaling enabled
    for calibrated probability estimation.
    """
    return SVC(
        kernel="linear",
        C=C,
        probability=True,
        random_state=random_state,
    )
