"""
WiFi Intelligence System — AI Analytics Engine
=================================================
Machine-learning models for WiFi behaviour analysis:

1. **Random Forest** — primary anomaly classifier
2. **Logistic Regression** — secondary / comparison classifier

Feature vector (per observation window):
    mean_rssi, std_rssi, min_rssi, max_rssi, rssi_range,
    mean_noise, snr, device_count, hour_of_day, day_of_week

Labels:
    0 = normal, 1 = anomaly

The module ships with a synthetic labelled dataset generator so the
dashboard can demonstrate ML predictions even without historical data.
"""

from __future__ import annotations

import datetime
import math
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from utils import DATASET_DIR, get_logger

log = get_logger("ai_analyzer")

MODEL_DIR = DATASET_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

FEATURES = [
    "mean_rssi", "std_rssi", "min_rssi", "max_rssi", "rssi_range",
    "mean_noise", "snr", "device_count", "hour_of_day", "day_of_week",
]

# ---------------------------------------------------------------------------
# Synthetic dataset generator
# ---------------------------------------------------------------------------

def generate_synthetic_dataset(n_samples: int = 2000, anomaly_ratio: float = 0.12) -> pd.DataFrame:
    """
    Generate a labelled dataset for training.

    Normal samples have:
        - RSSI around -50 to -65, low variance, 4-10 devices
    Anomalous samples have:
        - Sudden RSSI spikes/drops, very high or zero device counts,
          or unusual hours of unusual activity
    """
    rows = []
    n_anomaly = int(n_samples * anomaly_ratio)
    n_normal = n_samples - n_anomaly

    for _ in range(n_normal):
        mean_rssi = random.gauss(-57, 5)
        std_rssi = abs(random.gauss(2, 1))
        min_rssi = mean_rssi - random.uniform(3, 10)
        max_rssi = mean_rssi + random.uniform(1, 5)
        mean_noise = random.gauss(-95, 1)
        device_count = random.randint(3, 12)
        hour = random.choices(range(24), weights=[
            1,1,1,1,1,2,3,5,6,6,5,5,5,5,6,6,7,8,8,7,5,3,2,1
        ])[0]
        rows.append({
            "mean_rssi": round(mean_rssi, 2),
            "std_rssi": round(std_rssi, 2),
            "min_rssi": round(min_rssi, 2),
            "max_rssi": round(max_rssi, 2),
            "rssi_range": round(max_rssi - min_rssi, 2),
            "mean_noise": round(mean_noise, 2),
            "snr": round(mean_rssi - mean_noise, 2),
            "device_count": device_count,
            "hour_of_day": hour,
            "day_of_week": random.randint(0, 6),
            "label": 0,
        })

    for _ in range(n_anomaly):
        anomaly_type = random.choice(["signal_drop", "signal_spike", "device_flood", "device_vanish", "late_night"])
        if anomaly_type == "signal_drop":
            mean_rssi = random.gauss(-82, 4)
            std_rssi = abs(random.gauss(8, 3))
        elif anomaly_type == "signal_spike":
            mean_rssi = random.gauss(-30, 3)
            std_rssi = abs(random.gauss(6, 2))
        else:
            mean_rssi = random.gauss(-57, 5)
            std_rssi = abs(random.gauss(2, 1))

        min_rssi = mean_rssi - random.uniform(5, 20)
        max_rssi = mean_rssi + random.uniform(2, 10)
        mean_noise = random.gauss(-95, 1)

        if anomaly_type == "device_flood":
            device_count = random.randint(20, 50)
        elif anomaly_type == "device_vanish":
            device_count = random.randint(0, 1)
        else:
            device_count = random.randint(3, 12)

        if anomaly_type == "late_night":
            hour = random.choice([1, 2, 3, 4])
            device_count = random.randint(8, 20)
        else:
            hour = random.randint(0, 23)

        rows.append({
            "mean_rssi": round(mean_rssi, 2),
            "std_rssi": round(std_rssi, 2),
            "min_rssi": round(min_rssi, 2),
            "max_rssi": round(max_rssi, 2),
            "rssi_range": round(max_rssi - min_rssi, 2),
            "mean_noise": round(mean_noise, 2),
            "snr": round(mean_rssi - mean_noise, 2),
            "device_count": device_count,
            "hour_of_day": hour,
            "day_of_week": random.randint(0, 6),
            "label": 1,
        })

    df = pd.DataFrame(rows)
    return df.sample(frac=1, random_state=42).reset_index(drop=True)


# ---------------------------------------------------------------------------
# AI Analyzer
# ---------------------------------------------------------------------------

class AIAnalyzer:
    """
    Trains and runs anomaly detection models on WiFi network features.
    """

    def __init__(self):
        self.scaler = StandardScaler()
        self.rf_model = RandomForestClassifier(
            n_estimators=120, max_depth=10, random_state=42, n_jobs=-1,
        )
        self.lr_model = LogisticRegression(
            max_iter=1000, random_state=42,
        )
        self._is_trained = False
        self._metrics: Dict[str, Dict] = {}

    # -- training ----------------------------------------------------------

    def train(self, df: Optional[pd.DataFrame] = None):
        """
        Train both models.  If no DataFrame is supplied, a synthetic
        dataset is generated automatically.
        """
        if df is None:
            log.info("No dataset provided — generating synthetic training data")
            df = generate_synthetic_dataset()

        X = df[FEATURES].values
        y = df["label"].values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y,
        )

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Random Forest
        self.rf_model.fit(X_train_scaled, y_train)
        rf_pred = self.rf_model.predict(X_test_scaled)
        rf_acc = accuracy_score(y_test, rf_pred)
        self._metrics["random_forest"] = {
            "accuracy": round(rf_acc, 4),
            "report": classification_report(y_test, rf_pred, output_dict=True),
        }
        log.info("Random Forest accuracy: %.2f%%", rf_acc * 100)

        # Logistic Regression
        self.lr_model.fit(X_train_scaled, y_train)
        lr_pred = self.lr_model.predict(X_test_scaled)
        lr_acc = accuracy_score(y_test, lr_pred)
        self._metrics["logistic_regression"] = {
            "accuracy": round(lr_acc, 4),
            "report": classification_report(y_test, lr_pred, output_dict=True),
        }
        log.info("Logistic Regression accuracy: %.2f%%", lr_acc * 100)

        self._is_trained = True

    # -- prediction --------------------------------------------------------

    def predict(self, features: Dict | pd.DataFrame, model: str = "random_forest") -> np.ndarray:
        """
        Predict anomaly label(s) for one or more observations.

        Parameters
        ----------
        features : dict or DataFrame
            A single observation (dict) or multiple (DataFrame) with
            columns matching FEATURES.
        model : str
            'random_forest' or 'logistic_regression'
        """
        if not self._is_trained:
            self.train()

        if isinstance(features, dict):
            df = pd.DataFrame([features])
        else:
            df = features

        X = self.scaler.transform(df[FEATURES].values)
        clf = self.rf_model if model == "random_forest" else self.lr_model
        return clf.predict(X)

    def predict_proba(self, features: Dict | pd.DataFrame, model: str = "random_forest") -> np.ndarray:
        """Return class probabilities."""
        if not self._is_trained:
            self.train()
        if isinstance(features, dict):
            df = pd.DataFrame([features])
        else:
            df = features
        X = self.scaler.transform(df[FEATURES].values)
        clf = self.rf_model if model == "random_forest" else self.lr_model
        return clf.predict_proba(X)

    # -- feature extraction helpers ----------------------------------------

    @staticmethod
    def extract_features(
        rssi_values: List[float],
        noise_values: Optional[List[float]] = None,
        device_count: int = 0,
        timestamp: Optional[datetime.datetime] = None,
    ) -> Dict:
        """
        Build a feature dict from raw signal/device observations.
        Suitable for passing to predict().
        """
        arr = np.array(rssi_values) if rssi_values else np.array([-60])
        noise = np.array(noise_values) if noise_values else np.array([-95])
        ts = timestamp or datetime.datetime.now()

        return {
            "mean_rssi": round(float(arr.mean()), 2),
            "std_rssi": round(float(arr.std()), 2),
            "min_rssi": round(float(arr.min()), 2),
            "max_rssi": round(float(arr.max()), 2),
            "rssi_range": round(float(arr.max() - arr.min()), 2),
            "mean_noise": round(float(noise.mean()), 2),
            "snr": round(float(arr.mean() - noise.mean()), 2),
            "device_count": device_count,
            "hour_of_day": ts.hour,
            "day_of_week": ts.weekday(),
        }

    # -- model persistence -------------------------------------------------

    def save_models(self):
        joblib.dump(self.rf_model, MODEL_DIR / "rf_model.pkl")
        joblib.dump(self.lr_model, MODEL_DIR / "lr_model.pkl")
        joblib.dump(self.scaler, MODEL_DIR / "scaler.pkl")
        log.info("Models saved to %s", MODEL_DIR)

    def load_models(self) -> bool:
        try:
            self.rf_model = joblib.load(MODEL_DIR / "rf_model.pkl")
            self.lr_model = joblib.load(MODEL_DIR / "lr_model.pkl")
            self.scaler = joblib.load(MODEL_DIR / "scaler.pkl")
            self._is_trained = True
            log.info("Models loaded from %s", MODEL_DIR)
            return True
        except FileNotFoundError:
            return False

    # -- reporting ---------------------------------------------------------

    @property
    def is_trained(self) -> bool:
        return self._is_trained

    @property
    def metrics(self) -> Dict:
        return self._metrics

    def feature_importances(self) -> pd.DataFrame:
        """Return Random Forest feature importances as a sorted DataFrame."""
        if not self._is_trained:
            return pd.DataFrame()
        imp = self.rf_model.feature_importances_
        df = pd.DataFrame({"feature": FEATURES, "importance": imp})
        return df.sort_values("importance", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("🤖  Training AI Analyzer on synthetic data …")
    analyzer = AIAnalyzer()
    analyzer.train()

    print("\n📊  Random Forest metrics:")
    print(f"   Accuracy: {analyzer.metrics['random_forest']['accuracy']:.2%}")

    print("\n📊  Logistic Regression metrics:")
    print(f"   Accuracy: {analyzer.metrics['logistic_regression']['accuracy']:.2%}")

    print("\n🔍  Feature importances (Random Forest):")
    print(analyzer.feature_importances().to_string(index=False))

    # Test single prediction
    sample = AIAnalyzer.extract_features(
        rssi_values=[-55, -57, -53, -56],
        noise_values=[-95, -94, -95],
        device_count=7,
    )
    pred = analyzer.predict(sample)
    proba = analyzer.predict_proba(sample)
    print(f"\nSample prediction: {'⚠️  ANOMALY' if pred[0] == 1 else '✅  NORMAL'}")
    print(f"   Probabilities: Normal={proba[0][0]:.2%}  Anomaly={proba[0][1]:.2%}")

    analyzer.save_models()
    print("\n✅  ai_analyzer.py self-test passed")
