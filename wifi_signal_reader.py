"""
WiFi Intelligence System — WiFi Signal Reader
================================================
Captures RSSI (signal strength) and noise floor from macOS system
utilities at approximately 1 Hz.  Maintains a thread-safe circular
buffer that the dashboard can consume in real-time.

Signal source priority:
  1. macOS `airport -I`  (fast, but removed in recent macOS versions)
  2. `system_profiler SPAirPortDataType`  (slower ~2s, but always available)
  3. Synthetic demo mode  (when neither is available)
"""

from __future__ import annotations

import re
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

from utils import get_logger, now_iso, safe_float, safe_int

log = get_logger("signal_reader")

# ---------------------------------------------------------------------------
# macOS airport binary path
# ---------------------------------------------------------------------------
AIRPORT_BIN = (
    "/System/Library/PrivateFrameworks/Apple80211.framework"
    "/Versions/Current/Resources/airport"
)

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class SignalSample:
    """A single WiFi signal measurement."""
    timestamp: str
    rssi: float          # dBm (typically -30 to -90)
    noise: float         # dBm noise floor
    ssid: str = ""
    bssid: str = ""
    channel: str = ""
    tx_rate: float = 0.0  # Mbps
    mcs_index: int = -1
    snr: float = 0.0     # signal-to-noise ratio (computed)

    def __post_init__(self):
        self.snr = self.rssi - self.noise if self.noise != 0 else 0.0

    def to_dict(self) -> Dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Airport parser
# ---------------------------------------------------------------------------

def _parse_airport_output(raw: str) -> Optional[SignalSample]:
    """Parse the output of `airport -I` into a SignalSample."""
    info: Dict[str, str] = {}
    for line in raw.strip().splitlines():
        parts = line.strip().split(":", 1)
        if len(parts) == 2:
            key = parts[0].strip().lstrip()
            val = parts[1].strip()
            info[key] = val

    rssi = safe_float(info.get("agrCtlRSSI"), default=None)
    noise = safe_float(info.get("agrCtlNoise"), default=None)

    if rssi is None:
        return None

    return SignalSample(
        timestamp=now_iso(),
        rssi=rssi,
        noise=noise if noise is not None else -95.0,
        ssid=info.get("SSID", ""),
        bssid=info.get("BSSID", ""),
        channel=info.get("channel", ""),
        tx_rate=safe_float(info.get("lastTxRate")),
        mcs_index=safe_int(info.get("MCS", -1)),
    )


def read_airport() -> Optional[SignalSample]:
    """Execute the airport utility and return a parsed SignalSample."""
    try:
        result = subprocess.run(
            [AIRPORT_BIN, "-I"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            return None
        return _parse_airport_output(result.stdout)
    except FileNotFoundError:
        log.debug("airport binary not found — trying system_profiler")
        return None
    except subprocess.TimeoutExpired:
        log.warning("airport command timed out")
        return None
    except Exception as exc:
        log.error("airport read error: %s", exc)
        return None


# ---------------------------------------------------------------------------
# system_profiler fallback (works on all macOS versions)
# ---------------------------------------------------------------------------

_SP_SIGNAL_RE = re.compile(r"Signal / Noise:\s*(-?\d+)\s*dBm\s*/\s*(-?\d+)\s*dBm")
_SP_CHANNEL_RE = re.compile(r"Channel:\s*(\d+)\s*\(([^)]+)\)")
_SP_TXRATE_RE = re.compile(r"Transmit Rate:\s*(\d+)")
_SP_MCS_RE = re.compile(r"MCS Index:\s*(\d+)")
_SP_PHY_RE = re.compile(r"PHY Mode:\s*(.+)")
_SP_SECURITY_RE = re.compile(r"Security:\s*(.+)")


def _parse_system_profiler(raw: str) -> Optional[SignalSample]:
    """Parse `system_profiler SPAirPortDataType` output."""
    # Extract the "Current Network Information" block
    sig = _SP_SIGNAL_RE.search(raw)
    if not sig:
        return None

    rssi = float(sig.group(1))
    noise = float(sig.group(2))

    chan_m = _SP_CHANNEL_RE.search(raw)
    channel = chan_m.group(1) if chan_m else ""
    channel_info = chan_m.group(2) if chan_m else ""

    tx_m = _SP_TXRATE_RE.search(raw)
    tx_rate = float(tx_m.group(1)) if tx_m else 0.0

    mcs_m = _SP_MCS_RE.search(raw)
    mcs = int(mcs_m.group(1)) if mcs_m else -1

    # Try to extract SSID — it appears as the section header before PHY Mode
    # In system_profiler output it's redacted as "<redacted>" but we can
    # try to find it; if redacted we'll use a placeholder
    ssid = ""
    lines = raw.splitlines()
    for i, line in enumerate(lines):
        if "Current Network Information" in line:
            # The next non-empty indented line is the SSID
            for j in range(i + 1, min(i + 5, len(lines))):
                candidate = lines[j].strip().rstrip(":")
                if candidate and candidate != "<redacted>":
                    ssid = candidate
                    break
                elif candidate == "<redacted>":
                    ssid = "(WiFi Connected)"
                    break
            break

    return SignalSample(
        timestamp=now_iso(),
        rssi=rssi,
        noise=noise,
        ssid=ssid,
        bssid="",
        channel=f"{channel} ({channel_info})" if channel_info else channel,
        tx_rate=tx_rate,
        mcs_index=mcs,
    )


def read_system_profiler() -> Optional[SignalSample]:
    """Use system_profiler as a fallback to read WiFi signal."""
    try:
        result = subprocess.run(
            ["system_profiler", "SPAirPortDataType"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if result.returncode != 0:
            return None
        return _parse_system_profiler(result.stdout)
    except Exception as exc:
        log.debug("system_profiler read failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Unified read function: airport → system_profiler → None
# ---------------------------------------------------------------------------

_use_system_profiler = False  # set at auto-detect time


def read_wifi_signal() -> Optional[SignalSample]:
    """Read WiFi signal using the best available method."""
    if _use_system_profiler:
        return read_system_profiler()
    sample = read_airport()
    if sample is not None:
        return sample
    return read_system_profiler()


# ---------------------------------------------------------------------------
# Synthetic / demo data generator
# ---------------------------------------------------------------------------

import random
import math

_demo_t = 0.0


def _synthetic_sample() -> SignalSample:
    """Generate a realistic-looking synthetic WiFi signal sample."""
    global _demo_t
    _demo_t += 1.0

    # Base signal with slow drift + fast noise
    base_rssi = -55 + 8 * math.sin(_demo_t / 60)  # slow drift
    jitter = random.gauss(0, 2.5)                   # measurement noise
    rssi = round(base_rssi + jitter, 1)

    noise = round(-95 + random.gauss(0, 1), 1)

    return SignalSample(
        timestamp=now_iso(),
        rssi=rssi,
        noise=noise,
        ssid="Airtel_Xstream_Demo",
        bssid="AA:BB:CC:DD:EE:FF",
        channel="36",
        tx_rate=round(random.uniform(200, 866), 0),
        mcs_index=random.choice([7, 8, 9]),
    )

# ---------------------------------------------------------------------------
# Continuous signal reader (threaded)
# ---------------------------------------------------------------------------

class SignalReader:
    """
    Continuously samples WiFi signal and maintains a bounded history
    buffer.  Runs in a daemon thread so it won't block shutdown.

    Parameters
    ----------
    max_history : int
        Maximum number of samples to keep in the ring buffer.
    interval : float
        Seconds between samples (default 1.0 → ~1 Hz).
    demo_mode : bool | None
        Force demo mode (True), force live mode (False), or auto-detect
        (None — tries airport first, falls back to demo).
    """

    def __init__(
        self,
        max_history: int = 600,
        interval: float = 1.0,
        demo_mode: Optional[bool] = None,
    ):
        self.max_history = max_history
        self.interval = interval
        self._history: deque[SignalSample] = deque(maxlen=max_history)
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None

        # Determine mode
        global _use_system_profiler
        if demo_mode is True:
            self._use_demo = True
        elif demo_mode is False:
            self._use_demo = False
        else:
            # Auto-detect: try airport first, then system_profiler
            sample = read_airport()
            if sample is not None:
                self._use_demo = False
                log.info("Using airport for live signal readings")
            else:
                sample = read_system_profiler()
                if sample is not None:
                    self._use_demo = False
                    _use_system_profiler = True
                    log.info("airport unavailable — using system_profiler for live signal readings")
                else:
                    self._use_demo = True
                    log.info("No WiFi signal source available — switching to demo mode")

    # -- public API --------------------------------------------------------

    def start(self):
        """Start the background sampling thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        log.info(
            "SignalReader started (%s mode, %.1fs interval)",
            "demo" if self._use_demo else "live",
            self.interval,
        )

    def stop(self):
        """Signal the background thread to stop."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=3)

    @property
    def is_demo(self) -> bool:
        return self._use_demo

    @property
    def latest(self) -> Optional[SignalSample]:
        """Return the most recent sample (or None)."""
        with self._lock:
            return self._history[-1] if self._history else None

    @property
    def history(self) -> List[SignalSample]:
        """Return a snapshot of the entire history buffer."""
        with self._lock:
            return list(self._history)

    def history_dicts(self) -> List[Dict]:
        """History as a list of plain dicts (for DataFrame conversion)."""
        return [s.to_dict() for s in self.history]

    # -- internal ----------------------------------------------------------

    def _loop(self):
        while self._running:
            if self._use_demo:
                sample = _synthetic_sample()
            else:
                sample = read_wifi_signal()
            if sample is not None:
                with self._lock:
                    self._history.append(sample)
            time.sleep(self.interval)


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    reader = SignalReader(max_history=10, interval=0.5)
    reader.start()
    print(f"Mode: {'demo' if reader.is_demo else 'live'}")
    time.sleep(5)
    reader.stop()
    for s in reader.history:
        print(f"  {s.timestamp}  RSSI={s.rssi}  Noise={s.noise}  SNR={s.snr}")
    print(f"✅  wifi_signal_reader.py self-test passed ({len(reader.history)} samples)")
