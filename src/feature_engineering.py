"""
Feature Engineering & Fault Localization Module

Directly implements Equations (1) & (2) from Section III of the reference paper:
    x1_n = Ampgain_n - Targetgain_n
    x2_n = Target_span_loss_n - span_loss_n
    
    X1 = sum(x1_n) for n=0..18
    X2 = sum(x2_n) for n=0..18

Also implements span-level optical fault localization and root-cause diagnosis.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


def compute_span_level_differences(
    df: pd.DataFrame, n_spans: int = 19
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes per-span difference metrics x1_n and x2_n according to Eq. 1:
        x1_n = amplifier_gain_n - target_gain_n
        x2_n = target_span_loss_n - span_loss_n

    Returns:
    --------
    x1_matrix : np.ndarray, shape (N, n_spans)
        Amplifier gain differences.
    x2_matrix : np.ndarray, shape (N, n_spans)
        Span loss differences.
    """
    n_samples = len(df)
    x1_matrix = np.zeros((n_samples, n_spans), dtype=np.float64)
    x2_matrix = np.zeros((n_samples, n_spans), dtype=np.float64)

    for s in range(n_spans):
        amp_gain_col = f"amplifier_gain_span_{s}"
        target_gain_col = f"target_gain_span_{s}"
        span_loss_col = f"span_loss_span_{s}"
        target_span_loss_col = f"target_span_loss_span_{s}"

        if (
            amp_gain_col in df.columns
            and target_gain_col in df.columns
            and span_loss_col in df.columns
            and target_span_loss_col in df.columns
        ):
            x1_matrix[:, s] = df[amp_gain_col].values - df[target_gain_col].values
            x2_matrix[:, s] = df[target_span_loss_col].values - df[span_loss_col].values
        else:
            raise KeyError(
                f"Missing required span telemetry columns for span {s}."
            )

    return x1_matrix, x2_matrix


def aggregate_features(x1_matrix: np.ndarray, x2_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Aggregates span differences across all 19 spans according to Eq. 2:
        X1 = sum(x1_n) for n=0..18
        X2 = sum(x2_n) for n=0..18
    """
    X1 = np.sum(x1_matrix, axis=1)
    X2 = np.sum(x2_matrix, axis=1)
    return X1, X2


def extract_features_from_raw(raw_df: pd.DataFrame, n_spans: int = 19) -> pd.DataFrame:
    """
    Transforms raw 19-span telemetry into aggregated 2-feature DataFrame.
    """
    x1_mat, x2_mat = compute_span_level_differences(raw_df, n_spans=n_spans)
    X1, X2 = aggregate_features(x1_mat, x2_mat)

    feat_df = pd.DataFrame({"X1": X1, "X2": X2})

    if "label" in raw_df.columns:
        feat_df["label"] = raw_df["label"].values
    if "direction" in raw_df.columns:
        feat_df["direction"] = raw_df["direction"].values
    if "timestamp" in raw_df.columns:
        feat_df["timestamp"] = raw_df["timestamp"].values
    if "failing_span" in raw_df.columns:
        feat_df["failing_span"] = raw_df["failing_span"].values

    return feat_df


def localize_fault(
    span_losses: Optional[Union[List[float], np.ndarray]] = None,
    target_span_losses: Optional[Union[List[float], np.ndarray]] = None,
    amp_gains: Optional[Union[List[float], np.ndarray]] = None,
    target_gains: Optional[Union[List[float], np.ndarray]] = None,
    n_spans: int = 19,
) -> Dict[str, Any]:
    """
    Localizes the culprit failing span in an optical link based on per-span telemetry.

    Parameters:
    -----------
    span_losses : list/array of 19 floats
    target_span_losses : list/array of 19 floats
    amp_gains : list/array of 19 floats
    target_gains : list/array of 19 floats

    Returns:
    --------
    dict with:
        supported: bool
        failing_span: int or None
        anomaly_metric: str
        max_deviation_db: float
        diagnosis: str
        per_span_x1: list
        per_span_x2: list
    """
    if (
        span_losses is None
        or target_span_losses is None
        or amp_gains is None
        or target_gains is None
    ):
        return {
            "supported": False,
            "failing_span": None,
            "anomaly_metric": "None",
            "max_deviation_db": 0.0,
            "diagnosis": "Localization requires per-span telemetry (all 19 spans). Only aggregate link condition is supported.",
            "per_span_x1": [],
            "per_span_x2": [],
        }

    sl = np.asarray(span_losses, dtype=float)
    tsl = np.asarray(target_span_losses, dtype=float)
    ag = np.asarray(amp_gains, dtype=float)
    tg = np.asarray(target_gains, dtype=float)

    if len(sl) != n_spans or len(ag) != n_spans:
        return {
            "supported": False,
            "failing_span": None,
            "anomaly_metric": "Incomplete",
            "max_deviation_db": 0.0,
            "diagnosis": f"Provided {len(sl)} spans; exactly {n_spans} spans required.",
            "per_span_x1": [],
            "per_span_x2": [],
        }

    x1 = ag - tg  # amplifier gain difference
    x2 = tsl - sl  # span loss difference (negative when span loss exceeds target)

    # Anomaly scoring: measure magnitude of degradation
    # A negative x2 indicates excess loss (dB); negative x1 indicates gain deficit (dB)
    excess_loss = np.maximum(0.0, -x2)
    gain_deficit = np.maximum(0.0, -x1)
    combined_anomaly = excess_loss + gain_deficit

    culprit_span = int(np.argmax(combined_anomaly))
    max_dev = float(combined_anomaly[culprit_span])

    # Threshold for attributing a specific span failure (~2.0 dB anomaly)
    if max_dev < 1.5:
        diagnosis = "All 19 spans operate within nominal thermal tolerances. No localized physical failure detected."
        culprit_span = -1
        metric = "Nominal"
    else:
        if excess_loss[culprit_span] > gain_deficit[culprit_span]:
            metric = "Excess Span Loss (Fiber Issue)"
            diagnosis = (
                f"Span {culprit_span} exhibits severe optical attenuation (+{excess_loss[culprit_span]:.2f} dB over target). "
                f"Likely root cause: fiber macro-bend, connector contamination, splice degradation, or impending fiber cut."
            )
        else:
            metric = "Amplifier Gain Deficit"
            diagnosis = (
                f"Span {culprit_span} amplifier exhibits gain deficiency (-{gain_deficit[culprit_span]:.2f} dB under target). "
                f"Likely root cause: optical amplifier (EDFA) pump degradation or gain saturation."
            )

    return {
        "supported": True,
        "failing_span": culprit_span,
        "anomaly_metric": metric,
        "max_deviation_db": round(max_dev, 2),
        "diagnosis": diagnosis,
        "per_span_x1": [round(float(v), 3) for v in x1],
        "per_span_x2": [round(float(v), 3) for v in x2],
    }
