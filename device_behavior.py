"""
WiFi Intelligence System — Device Behavior Analyzer
======================================================
Tracks device connection history over time and provides analytics:

* Active vs. inactive device classification
* Most-active device identification
* Peak usage time detection
* Connection frequency per device
* Historical time-series stored as CSV
"""

from __future__ import annotations

import csv
import datetime
import os
import random
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from network_scanner import NetworkDevice
from utils import DATASET_DIR, get_logger, now_iso

log = get_logger("device_behavior")

HISTORY_CSV = DATASET_DIR / "device_history.csv"

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class DeviceRecord:
    """Aggregated record for a single device across multiple scans."""
    mac: str
    ip: str = ""
    hostname: str = ""
    vendor: str = "Unknown"
    first_seen: str = ""
    last_seen: str = ""
    seen_count: int = 0
    active_count: int = 0
    inactive_count: int = 0
    total_minutes_online: float = 0.0

    @property
    def activity_ratio(self) -> float:
        return self.active_count / max(self.seen_count, 1)

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["activity_ratio"] = round(self.activity_ratio, 2)
        return d


# ---------------------------------------------------------------------------
# Behavior tracker
# ---------------------------------------------------------------------------

class DeviceBehaviorTracker:
    """
    Maintains a running ledger of device observations and provides
    analytics queries.
    """

    def __init__(self):
        self._records: Dict[str, DeviceRecord] = {}
        self._scan_log: List[Dict] = []  # raw scan events
        self._load_history()

    # -- persistence -------------------------------------------------------

    def _load_history(self):
        """Load previous device history from CSV if it exists."""
        if not HISTORY_CSV.exists():
            return
        try:
            df = pd.read_csv(HISTORY_CSV)
            for _, row in df.iterrows():
                mac = row.get("mac", "")
                if not mac:
                    continue
                self._records[mac] = DeviceRecord(
                    mac=mac,
                    ip=row.get("ip", ""),
                    hostname=row.get("hostname", ""),
                    vendor=row.get("vendor", "Unknown"),
                    first_seen=row.get("first_seen", ""),
                    last_seen=row.get("last_seen", ""),
                    seen_count=int(row.get("seen_count", 0)),
                    active_count=int(row.get("active_count", 0)),
                    inactive_count=int(row.get("inactive_count", 0)),
                    total_minutes_online=float(row.get("total_minutes_online", 0)),
                )
            log.info("Loaded %d device records from history", len(self._records))
        except Exception as exc:
            log.warning("Could not load history: %s", exc)

    def save_history(self):
        """Persist the current device records to CSV."""
        rows = [r.to_dict() for r in self._records.values()]
        if not rows:
            return
        df = pd.DataFrame(rows)
        df.to_csv(HISTORY_CSV, index=False)
        log.debug("Saved %d records to %s", len(rows), HISTORY_CSV)

    # -- observation -------------------------------------------------------

    def observe(self, devices: List[NetworkDevice]):
        """
        Record a new scan observation.  Call this each time the network
        scanner returns a fresh device list.
        """
        ts = now_iso()
        for dev in devices:
            mac = dev.mac
            rec = self._records.get(mac)
            if rec is None:
                rec = DeviceRecord(
                    mac=mac,
                    ip=dev.ip,
                    hostname=dev.hostname,
                    vendor=dev.vendor,
                    first_seen=ts,
                )
                self._records[mac] = rec

            rec.ip = dev.ip
            rec.hostname = dev.hostname or rec.hostname
            rec.vendor = dev.vendor if dev.vendor != "Unknown" else rec.vendor
            rec.last_seen = ts
            rec.seen_count += 1
            if dev.is_active:
                rec.active_count += 1
            else:
                rec.inactive_count += 1

            # Approximate online minutes (assume scan interval ≈ 1 min)
            if dev.is_active:
                rec.total_minutes_online += 1.0

        # Log raw event
        self._scan_log.append({
            "timestamp": ts,
            "device_count": len(devices),
            "active_count": sum(1 for d in devices if d.is_active),
        })

    # -- analytics ---------------------------------------------------------

    @property
    def all_records(self) -> List[DeviceRecord]:
        return list(self._records.values())

    @property
    def device_count(self) -> int:
        return len(self._records)

    @property
    def active_devices(self) -> List[DeviceRecord]:
        """Devices seen as active in the most recent observation."""
        if not self._scan_log:
            return []
        latest_ts = self._scan_log[-1]["timestamp"]
        return [r for r in self._records.values() if r.last_seen == latest_ts and r.active_count > 0]

    def most_active_device(self) -> Optional[DeviceRecord]:
        """Device with the highest total online minutes."""
        if not self._records:
            return None
        return max(self._records.values(), key=lambda r: r.total_minutes_online)

    def peak_usage_time(self) -> Optional[str]:
        """Hour with the most scan events (proxy for peak usage)."""
        if not self._scan_log:
            return None
        hour_counts: Dict[int, int] = defaultdict(int)
        for entry in self._scan_log:
            try:
                h = datetime.datetime.fromisoformat(entry["timestamp"]).hour
                hour_counts[h] += entry["active_count"]
            except Exception:
                pass
        if not hour_counts:
            return None
        peak_h = max(hour_counts, key=hour_counts.get)
        return f"{peak_h:02d}:00 – {peak_h:02d}:59"

    def connection_frequency(self) -> pd.DataFrame:
        """DataFrame with (mac, hostname, vendor, seen_count, activity_ratio)."""
        rows = [r.to_dict() for r in self._records.values()]
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        return df[["mac", "hostname", "vendor", "seen_count", "active_count", "activity_ratio"]].sort_values(
            "active_count", ascending=False
        )

    def scan_timeline(self) -> pd.DataFrame:
        """DataFrame of scan events over time."""
        if not self._scan_log:
            return pd.DataFrame(columns=["timestamp", "device_count", "active_count"])
        return pd.DataFrame(self._scan_log)

    # -- summary -----------------------------------------------------------

    def summary_dict(self) -> Dict:
        most_active = self.most_active_device()
        return {
            "total_devices_seen": self.device_count,
            "currently_active": len(self.active_devices),
            "most_active_device": most_active.hostname or most_active.mac if most_active else "N/A",
            "most_active_vendor": most_active.vendor if most_active else "N/A",
            "peak_usage_time": self.peak_usage_time() or "N/A",
            "total_scans": len(self._scan_log),
        }


# ---------------------------------------------------------------------------
# Generate synthetic history for demo / testing
# ---------------------------------------------------------------------------

def generate_demo_history(hours: int = 24, scans_per_hour: int = 4) -> DeviceBehaviorTracker:
    """Create a tracker pre-loaded with synthetic multi-hour history."""
    from network_scanner import scan_demo

    tracker = DeviceBehaviorTracker()
    base_time = datetime.datetime.now() - datetime.timedelta(hours=hours)

    for h in range(hours):
        for s in range(scans_per_hour):
            # Advance time
            fake_now = base_time + datetime.timedelta(hours=h, minutes=s * 15)
            devices = scan_demo()
            # Patch timestamps
            ts = fake_now.isoformat(timespec="seconds")
            for d in devices:
                d.first_seen = ts
                d.last_seen = ts
            tracker.observe(devices)
            # Overwrite timestamp in scan_log for realism
            tracker._scan_log[-1]["timestamp"] = ts

    return tracker


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    tracker = generate_demo_history(hours=6)
    summary = tracker.summary_dict()
    print("\n📊  Device Behavior Summary:")
    for k, v in summary.items():
        print(f"   {k}: {v}")
    print("\n📋  Connection Frequency:")
    print(tracker.connection_frequency().to_string(index=False))
    tracker.save_history()
    print(f"\n💾  History saved to {HISTORY_CSV}")
    print("✅  device_behavior.py self-test passed")
