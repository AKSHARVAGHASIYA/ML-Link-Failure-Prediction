"""
Data Loader & Benchmark Generator Module

Faithfully implements the optical network telemetry schema specified in
Section III of "Link Failure Prediction and Localization in Cloud Scale
Networks Using Supervised Learning" (Bakhtiari, Stanford CS229).

Schema:
- 19 spans (n = 0..18) per end-to-end optical link
- 4 raw telemetry metrics per span:
    1. span_loss (dB)
    2. target_span_loss (dB)
    3. amplifier_gain (dB)
    4. target_gain (dB)
- Traffic direction: 'EW' (East-West) or 'WE' (West-East)
- 6,000 total time-series observations in 2-minute bins
- Ground truth binary labels: 0 = Healthy, 1 = Failing / Marginal
"""

from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import pandas as pd


def generate_benchmark_optical_dataset(
    n_samples: int = 6000,
    n_spans: int = 19,
    failure_ratio: float = 0.25,
    missing_rate: float = 0.015,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate synthetic benchmark optical telemetry following the paper's physical model.

    Parameters:
    -----------
    n_samples : int
        Total observations (paper uses 6,000).
    n_spans : int
        Number of optical spans per E2E link (paper uses 19).
    failure_ratio : float
        Proportion of degraded/failing links (default 0.25).
    missing_rate : float
        Fraction of missing telemetry cells to simulate network dropouts (default 1.5%).
    random_state : int
        Seed for reproducibility.

    Returns:
    --------
    raw_df : pd.DataFrame
        Detailed DataFrame with 19 spans x 4 metrics + metadata.
    features_df : pd.DataFrame
        Aggregated features (X1, X2, label).
    """
    rng = np.random.default_rng(random_state)

    # 1. Base network configuration
    nominal_span_loss = 16.0  # dB for standard 80 km SMF span at 0.2 dB/km
    nominal_target_gain = 16.0  # dB nominal EDFA compensation

    # Directions: EW (East-West) and WE (West-East)
    directions = rng.choice(["EW", "WE"], size=n_samples, p=[0.5, 0.5])
    timestamps = pd.date_range(start="2026-01-01 00:00:00", periods=n_samples, freq="2min")

    # 2. Determine class labels (0 = Healthy, 1 = Failing / Marginal)
    n_failures = int(n_samples * failure_ratio)
    labels = np.zeros(n_samples, dtype=int)
    labels[:n_failures] = 1
    rng.shuffle(labels)

    raw_data = {
        "timestamp": timestamps,
        "direction": directions,
        "label": labels,
    }

    # Arrays to store span-level differences for aggregation
    x1_spans = np.zeros((n_samples, n_spans))  # Ampgain - Targetgain
    x2_spans = np.zeros((n_samples, n_spans))  # Target_span_loss - span_loss

    failing_spans = []

    for i in range(n_samples):
        is_failing = labels[i] == 1
        
        # In healthy condition: slight fluctuations around nominal values
        span_losses = nominal_span_loss + rng.normal(0.0, 0.35, size=n_spans)
        target_span_losses = np.full(n_spans, nominal_span_loss)

        target_gains = np.full(n_spans, nominal_target_gain)
        # EDFAs track target gain with minor thermal drift
        amp_gains = nominal_target_gain + rng.normal(0.0, 0.30, size=n_spans)

        fault_span_idx = -1
        if is_failing:
            # Pick 1 or 2 spans exhibiting physical degradation (soft failure or fiber cut)
            fault_span_idx = int(rng.integers(0, n_spans))
            anomaly_type = rng.choice(["fiber_attenuation", "amplifier_deficit", "joint_degradation"])

            if anomaly_type == "fiber_attenuation":
                # Macro-bend, connector dirt, or pinch: span loss increases by 3 - 9 dB
                added_loss = rng.uniform(3.5, 9.0)
                span_losses[fault_span_idx] += added_loss
                # Downstream amplifier saturates partially
                amp_gains[fault_span_idx] -= rng.uniform(1.0, 3.5)

            elif anomaly_type == "amplifier_deficit":
                # EDFA pump degradation: amplifier fails to meet target gain
                deficit = rng.uniform(4.0, 8.5)
                amp_gains[fault_span_idx] -= deficit
                span_losses[fault_span_idx] += rng.uniform(0.5, 1.5)

            else:  # joint_degradation
                added_loss = rng.uniform(3.0, 6.0)
                span_losses[fault_span_idx] += added_loss
                amp_gains[fault_span_idx] -= rng.uniform(2.5, 5.0)

        failing_spans.append(fault_span_idx)

        # Store span data
        for s in range(n_spans):
            x1_spans[i, s] = amp_gains[s] - target_gains[s]
            x2_spans[i, s] = target_span_losses[s] - span_losses[s]

            raw_data.setdefault(f"target_span_loss_span_{s}", []).append(target_span_losses[s])
            raw_data.setdefault(f"span_loss_span_{s}", []).append(span_losses[s])
            raw_data.setdefault(f"target_gain_span_{s}", []).append(target_gains[s])
            raw_data.setdefault(f"amplifier_gain_span_{s}", []).append(amp_gains[s])

    raw_data["failing_span"] = failing_spans
    raw_df = pd.DataFrame(raw_data)

    # 3. Simulate missing measurements (random dropouts)
    if missing_rate > 0:
        telemetry_cols = [c for c in raw_df.columns if c.startswith(("span_loss_", "amplifier_gain_"))]
        total_telemetry_cells = len(raw_df) * len(telemetry_cols)
        n_missing = int(total_telemetry_cells * missing_rate)

        missing_rows = rng.integers(0, len(raw_df), size=n_missing)
        missing_cols = rng.choice(telemetry_cols, size=n_missing)

        for r, c in zip(missing_rows, missing_cols):
            raw_df.at[r, c] = np.nan

    # 4. Compute aggregated features per Eq. 2:
    # X1 = sum(x1_n), X2 = sum(x2_n)
    X1 = x1_spans.sum(axis=1)
    X2 = x2_spans.sum(axis=1)

    features_df = pd.DataFrame({
        "timestamp": timestamps,
        "direction": directions,
        "X1": X1,
        "X2": X2,
        "label": labels,
        "failing_span": failing_spans,
    })

    return raw_df, features_df


def load_or_generate_dataset(
    data_dir: str = "data",
    force_generate: bool = False,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load existing optical network dataset if available; otherwise generate
    and persist the benchmark dataset adhering to the paper's specification.

    Returns:
    --------
    raw_df : pd.DataFrame
        Raw telemetry per span.
    features_df : pd.DataFrame
        Engineered X1, X2 features and target labels.
    """
    dir_path = Path(data_dir)
    dir_path.mkdir(parents=True, exist_ok=True)

    raw_csv = dir_path / "optical_network_raw.csv"
    features_csv = dir_path / "optical_network_features.csv"

    if raw_csv.exists() and features_csv.exists() and not force_generate:
        raw_df = pd.read_csv(raw_csv)
        features_df = pd.read_csv(features_csv)
        if "timestamp" in raw_df.columns:
            raw_df["timestamp"] = pd.to_datetime(raw_df["timestamp"])
        if "timestamp" in features_df.columns:
            features_df["timestamp"] = pd.to_datetime(features_df["timestamp"])
        return raw_df, features_df

    # Generate dataset
    raw_df, features_df = generate_benchmark_optical_dataset(
        n_samples=6000,
        n_spans=19,
        failure_ratio=0.25,
        missing_rate=0.015,
        random_state=random_state,
    )

    # Persist to disk
    raw_df.to_csv(raw_csv, index=False)
    features_df.to_csv(features_csv, index=False)

    return raw_df, features_df


if __name__ == "__main__":
    raw_df, features_df = load_or_generate_dataset()
    print(f"Generated raw dataset shape: {raw_df.shape}")
    print(f"Generated features dataset shape: {features_df.shape}")
    print(f"Class distribution:\n{features_df['label'].value_counts(normalize=True)}")
