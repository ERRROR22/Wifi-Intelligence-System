"""
WiFi Intelligence System — Shared Utilities
=============================================
Centralised helpers used across every module: logging, paths, MAC-vendor
resolution, and timestamp formatting.
"""

import os
import logging
import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_DIR = PROJECT_ROOT / "dataset"
DATASET_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a consistently-formatted logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = logging.Formatter(
            "[%(asctime)s] %(name)s — %(levelname)s — %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


log = get_logger("wifi_intel")

# ---------------------------------------------------------------------------
# Timestamps
# ---------------------------------------------------------------------------

def now_iso() -> str:
    """Current local time as ISO-8601 string."""
    return datetime.datetime.now().isoformat(timespec="seconds")


def now_epoch() -> float:
    """Current time as Unix epoch (seconds)."""
    return datetime.datetime.now().timestamp()

# ---------------------------------------------------------------------------
# MAC → Vendor lookup
# ---------------------------------------------------------------------------

_mac_lookup = None


def _init_mac_lookup():
    """Lazy-load the MAC vendor database (downloads ~1 MB on first use)."""
    global _mac_lookup
    if _mac_lookup is None:
        try:
            from mac_vendor_lookup import MacLookup
            _mac_lookup = MacLookup()
            try:
                _mac_lookup.update_vendors()  # refresh OUI list
            except Exception:
                pass  # use cached version if offline
        except ImportError:
            log.warning("mac-vendor-lookup not installed — vendor resolution disabled")
            _mac_lookup = False  # sentinel so we don't retry


def mac_to_vendor(mac: str) -> str:
    """
    Resolve a MAC address to its vendor/manufacturer string.
    Returns 'Unknown' on failure.
    """
    _init_mac_lookup()
    if _mac_lookup is False:
        return "Unknown"
    try:
        return _mac_lookup.lookup(mac)
    except Exception:
        return "Unknown"

# ---------------------------------------------------------------------------
# Misc helpers
# ---------------------------------------------------------------------------

def safe_float(value, default: float = 0.0) -> float:
    """Convert *value* to float, returning *default* on failure."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_int(value, default: int = 0) -> int:
    """Convert *value* to int, returning *default* on failure."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    log.info("Project root : %s", PROJECT_ROOT)
    log.info("Dataset dir  : %s", DATASET_DIR)
    log.info("Current ISO  : %s", now_iso())
    log.info("MAC vendor   : %s", mac_to_vendor("00:14:22:01:23:45"))
    print("✅  utils.py self-test passed")
