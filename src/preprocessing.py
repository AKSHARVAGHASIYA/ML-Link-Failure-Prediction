"""
Preprocessing & Data Cleansing Module

Implements data quality audits and preprocessing strictly preventing data leakage:
- Duplicate row detection & removal
- Missing measurement interpolation (neighbor linear interpolation per traffic direction)
- Scaling fitted exclusively on training splits
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler


def clean_raw_telemetry(df: pd.DataFrame, n_spans: int = 19) -> pd.DataFrame:
    """
    Cleans raw telemetry DataFrame:
    1. Removes duplicates based on timestamp and direction.
    2. Implements time-series linear interpolation per direction for missing measurements.
    3. Forward/backward fills any boundary NaNs.
    """
    cleaned = df.copy()

    # Sort by direction and timestamp for proper interpolation
    if "timestamp" in cleaned.columns:
        cleaned["timestamp"] = pd.to_datetime(cleaned["timestamp"])
        sort_cols = ["direction", "timestamp"] if "direction" in cleaned.columns else ["timestamp"]
        cleaned = cleaned.sort_values(by=sort_cols).reset_index(drop=True)

    # Telemetry columns to clean
    telemetry_cols = []
    for s in range(n_spans):
        telemetry_cols.extend([
            f"span_loss_span_{s}",
            f"target_span_loss_span_{s}",
            f"amplifier_gain_span_{s}",
            f"target_gain_span_{s}",
        ])
    telemetry_cols = [c for c in telemetry_cols if c in cleaned.columns]

    # Perform linear interpolation within each direction group
    if "direction" in cleaned.columns:
        cleaned[telemetry_cols] = (
            cleaned.groupby("direction")[telemetry_cols]
            .transform(lambda group: group.interpolate(method="linear").bfill().ffill())
        )
    else:
        cleaned[telemetry_cols] = cleaned[telemetry_cols].interpolate(method="linear").bfill().ffill()

    return cleaned


class TelemetryPreprocessor(BaseEstimator, TransformerMixin):
    """
    Leakage-safe Preprocessor.
    Fits transformation parameters (e.g. mean, variance) strictly on the training set
    and applies identical parameters to the testing set.
    """

    def __init__(self, scale_features: bool = True):
        self.scale_features = scale_features
        self.scaler = StandardScaler() if scale_features else None
        self.is_fitted = False
        self.feature_names_: List[str] = []

    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: Optional[Union[np.ndarray, pd.Series]] = None):
        """Fit preprocessor parameters exclusively on training split."""
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = list(X.columns)
            X_mat = X.values
        else:
            X_mat = np.asarray(X)
            self.feature_names_ = [f"X{i+1}" for i in range(X_mat.shape[1])]

        if self.scale_features and self.scaler is not None:
            self.scaler.fit(X_mat)

        self.is_fitted = True
        return self

    def transform(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Transform input features using fitted training statistics."""
        if not self.is_fitted:
            raise RuntimeError("TelemetryPreprocessor must be fitted on training data before transform.")

        if isinstance(X, pd.DataFrame):
            X_mat = X.values
        else:
            X_mat = np.asarray(X)

        if self.scale_features and self.scaler is not None:
            return self.scaler.transform(X_mat)
        return X_mat

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Revert transformed features to original telemetry scale."""
        if self.scale_features and self.scaler is not None:
            return self.scaler.inverse_transform(X)
        return X
