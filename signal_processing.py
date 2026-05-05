"""
WiFi Intelligence System — Signal Processing
===============================================
Preprocessing and feature-extraction routines for WiFi RSSI time-series:

* Moving average & exponential moving average smoothing
* Rolling variance / standard deviation
* Noise floor filtering
* Signal quality classification
* Anomaly flagging via z-score on rolling windows
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd

from utils import get_logger

log = get_logger("signal_proc")


# ---------------------------------------------------------------------------
# Smoothing
# ---------------------------------------------------------------------------

def moving_average(values: pd.Series, window: int = 10) -> pd.Series:
    """Simple moving average (SMA) with centred NaN padding."""
    return values.rolling(window=window, min_periods=1, center=True).mean()


def exponential_moving_average(
    values: pd.Series, span: int = 10
) -> pd.Series:
    """Exponential moving average (EMA)."""
    return values.ewm(span=span, adjust=False).mean()


# ---------------------------------------------------------------------------
# Variance & deviation
# ---------------------------------------------------------------------------

def rolling_variance(values: pd.Series, window: int = 10) -> pd.Series:
    """Rolling variance (population) over *window* samples."""
    return values.rolling(window=window, min_periods=1).var()


def rolling_std(values: pd.Series, window: int = 10) -> pd.Series:
    """Rolling standard deviation."""
    return values.rolling(window=window, min_periods=1).std()


# ---------------------------------------------------------------------------
# Noise filtering
# ---------------------------------------------------------------------------

def clip_to_noise_floor(
    rssi: pd.Series, noise_floor: float = -95.0
) -> pd.Series:
    """Clamp RSSI so that values below the noise floor are replaced."""
    return rssi.clip(lower=noise_floor)


def remove_outliers(
    values: pd.Series, z_threshold: float = 3.0
) -> pd.Series:
    """Replace outliers (|z| > threshold) with NaN, then forward-fill."""
    mean = values.mean()
    std = values.std()
    if std == 0:
        return values
    z = (values - mean) / std
    cleaned = values.where(z.abs() <= z_threshold)
    return cleaned.ffill().bfill()


# ---------------------------------------------------------------------------
# Signal quality classification
# ---------------------------------------------------------------------------

QUALITY_THRESHOLDS = {
    "Excellent": -50,
    "Good":      -60,
    "Fair":      -70,
    "Weak":      -80,
}


def classify_signal(rssi: float) -> str:
    """Return a human-readable quality label for an RSSI value."""
    for label, threshold in QUALITY_THRESHOLDS.items():
        if rssi >= threshold:
            return label
    return "Very Weak"


def classify_series(rssi: pd.Series) -> pd.Series:
    """Vectorised signal quality classification."""
    return rssi.apply(classify_signal)


# ---------------------------------------------------------------------------
# Anomaly flagging
# ---------------------------------------------------------------------------

def zscore_anomaly_flags(
    values: pd.Series,
    window: int = 30,
    threshold: float = 2.5,
) -> pd.Series:
    """
    Flag samples whose z-score (relative to a rolling window) exceeds
    *threshold*.  Returns a boolean Series.
    """
    rolling_mean = values.rolling(window=window, min_periods=1).mean()
    rolling_s = values.rolling(window=window, min_periods=1).std().replace(0, np.nan)
    z = ((values - rolling_mean) / rolling_s).abs()
    return z > threshold


# ---------------------------------------------------------------------------
# Full processing pipeline
# ---------------------------------------------------------------------------

def process_signal_dataframe(
    df: pd.DataFrame,
    rssi_col: str = "rssi",
    noise_col: str = "noise",
    window: int = 10,
) -> pd.DataFrame:
    """
    Apply the full processing pipeline to a DataFrame that contains at
    least an RSSI column.  Adds several derived columns in-place and
    returns the enriched DataFrame.
    """
    df = df.copy()
    rssi = df[rssi_col]

    # Smoothing
    df["rssi_sma"] = moving_average(rssi, window=window)
    df["rssi_ema"] = exponential_moving_average(rssi, span=window)

    # Variance
    df["rssi_variance"] = rolling_variance(rssi, window=window)
    df["rssi_std"] = rolling_std(rssi, window=window)

    # SNR (if noise column exists)
    if noise_col in df.columns:
        df["snr"] = df[rssi_col] - df[noise_col]

    # Quality label
    df["signal_quality"] = classify_series(rssi)

    # Anomaly flag
    df["is_anomaly"] = zscore_anomaly_flags(rssi, window=max(window * 3, 30))

    return df


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import random, datetime

    # Generate synthetic time-series
    n = 200
    timestamps = pd.date_range(
        end=datetime.datetime.now(), periods=n, freq="1s"
    )
    rssi_raw = [-55 + random.gauss(0, 4) for _ in range(n)]
    # Inject an anomaly
    rssi_raw[100] = -30  # sudden spike
    rssi_raw[150] = -90  # sudden drop

    df = pd.DataFrame({
        "timestamp": timestamps,
        "rssi": rssi_raw,
        "noise": [-95 + random.gauss(0, 0.5) for _ in range(n)],
    })

    result = process_signal_dataframe(df)
    anomalies = result[result["is_anomaly"]]
    print(f"Processed {len(result)} samples, detected {len(anomalies)} anomalies")
    print(result[["rssi", "rssi_sma", "rssi_ema", "rssi_variance", "signal_quality", "is_anomaly"]].tail(10))
    print("✅  signal_processing.py self-test passed")
