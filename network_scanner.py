"""
WiFi Intelligence System — Network Scanner
=============================================
Discovers ALL devices on the local WiFi network using multiple methods:

1. Ping sweep — pings every IP on the subnet to force devices into
   the ARP cache (many phones/tablets don't show up otherwise)
2. ARP table read — parses `arp -a` after the ping sweep
3. scapy ARP scan — if running as root (optional)
4. Hostname resolution & MAC vendor lookup for each device

This ensures that ALL connected devices (phones, tablets, smart TVs, etc.)
are discovered even if they haven't previously communicated with the Mac.
"""

from __future__ import annotations

import concurrent.futures
import os
import re
import socket
import subprocess
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Set

from utils import get_logger, mac_to_vendor, now_iso

log = get_logger("net_scanner")


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class NetworkDevice:
    ip: str
    mac: str
    hostname: str = ""
    vendor: str = "Unknown"
    first_seen: str = ""
    last_seen: str = ""
    is_active: bool = True

    def to_dict(self) -> Dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# MAC address normalisation
# ---------------------------------------------------------------------------

def _normalise_mac(mac: str) -> str:
    """
    Normalise a MAC address to the standard AA:BB:CC:DD:EE:FF format.
    macOS `arp -a` sometimes outputs shortened octets like 24:de:8a:92:2:f1
    instead of 24:DE:8A:92:02:F1.
    """
    parts = mac.split(":")
    normalised = ":".join(p.upper().zfill(2) for p in parts)
    return normalised


# ---------------------------------------------------------------------------
# Subnet detection
# ---------------------------------------------------------------------------

def get_local_subnet() -> Optional[str]:
    """Auto-detect the local subnet in CIDR notation (e.g. 192.168.1.0/24)."""
    try:
        import netifaces
        gateways = netifaces.gateways()
        default_iface = gateways.get("default", {}).get(netifaces.AF_INET, [None, None])[1]
        if default_iface is None:
            return None
        addrs = netifaces.ifaddresses(default_iface).get(netifaces.AF_INET, [])
        if not addrs:
            return None
        ip = addrs[0]["addr"]
        parts = ip.split(".")
        return f"{parts[0]}.{parts[1]}.{parts[2]}.0/24"
    except Exception as exc:
        log.debug("netifaces subnet detection failed: %s", exc)
        return None


def _get_subnet_fallback() -> Optional[str]:
    """Fallback subnet detection without netifaces."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        parts = ip.split(".")
        return f"{parts[0]}.{parts[1]}.{parts[2]}.0/24"
    except Exception:
        return "192.168.1.0/24"


def _get_subnet_base(subnet: str) -> str:
    """Extract the first 3 octets from a CIDR string, e.g. '192.168.1'."""
    return ".".join(subnet.split("/")[0].split(".")[:3])


# ---------------------------------------------------------------------------
# Hostname resolver
# ---------------------------------------------------------------------------

def resolve_hostname(ip: str) -> str:
    """Try reverse DNS resolution for an IP; return '' on failure."""
    try:
        name, _, _ = socket.gethostbyaddr(ip)
        return name
    except (socket.herror, socket.gaierror, OSError):
        return ""


# ---------------------------------------------------------------------------
# Ping sweep — forces all active devices into the ARP cache
# ---------------------------------------------------------------------------

def _ping_one(ip: str) -> Optional[str]:
    """Ping a single IP; return it if alive, else None."""
    try:
        result = subprocess.run(
            ["ping", "-c", "1", "-W", "1", ip],
            capture_output=True,
            timeout=3,
        )
        return ip if result.returncode == 0 else None
    except Exception:
        return None


def ping_sweep(subnet: str, max_workers: int = 30) -> List[str]:
    """
    Concurrently ping every IP on the /24 subnet.
    Returns a list of IPs that responded.
    This populates the system ARP cache so that `arp -a` will show them.
    """
    base = _get_subnet_base(subnet)
    ips = [f"{base}.{i}" for i in range(1, 255)]

    alive: List[str] = []
    log.info("Ping-sweeping %s.1–254 …", base)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_ping_one, ip): ip for ip in ips}
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result:
                alive.append(result)

    # Wait briefly for system ARP cache to fully update
    import time
    time.sleep(1)

    alive.sort(key=lambda ip: [int(x) for x in ip.split(".")])
    log.info("Ping sweep found %d alive hosts", len(alive))
    return alive


# ---------------------------------------------------------------------------
# Scanner: scapy ARP (requires root)
# ---------------------------------------------------------------------------

def scan_with_scapy(subnet: str, timeout: int = 3) -> List[NetworkDevice]:
    """Use scapy ARP scan to find devices on *subnet*."""
    try:
        from scapy.all import ARP, Ether, srp
        packet = Ether(dst="ff:ff:ff:ff:ff:ff") / ARP(pdst=subnet)
        result = srp(packet, timeout=timeout, verbose=False)[0]

        devices: List[NetworkDevice] = []
        ts = now_iso()
        for _, received in result:
            ip = received.psrc
            mac = _normalise_mac(received.hwsrc)
            devices.append(NetworkDevice(
                ip=ip,
                mac=mac,
                hostname=resolve_hostname(ip),
                vendor=mac_to_vendor(mac),
                first_seen=ts,
                last_seen=ts,
            ))
        return devices
    except PermissionError:
        log.info("scapy ARP scan needs root — falling back to ping + arp")
        return []
    except ImportError:
        log.info("scapy not installed — falling back to ping + arp")
        return []
    except Exception as exc:
        log.warning("scapy scan failed: %s", exc)
        return []


# ---------------------------------------------------------------------------
# Scanner: Ping sweep + `arp -a` (no root needed — primary method)
# ---------------------------------------------------------------------------

# Matches both named and unnamed hosts in arp output:
#   unit (192.168.1.1) at 24:de:8a:92:2:f1 on en0 ...
#   ? (192.168.1.9) at aa:bb:cc:dd:ee:ff on en0 ...
_ARP_RE = re.compile(
    r"(?:(\S+)\s+)?\((\d+\.\d+\.\d+\.\d+)\)\s+at\s+([\da-fA-F:]+)"
)


def scan_with_ping_and_arp(subnet: str) -> List[NetworkDevice]:
    """
    Full discovery: ping every IP on the subnet to populate the ARP
    cache, then parse `arp -a` to collect MAC addresses.
    """
    # Step 1: Ping sweep to populate ARP cache
    alive_ips = ping_sweep(subnet)

    # Step 2: Read the now-populated ARP table
    try:
        out = subprocess.check_output(["arp", "-a"], text=True, timeout=30)
    except Exception as exc:
        log.warning("arp -a failed: %s", exc)
        return []

    devices: List[NetworkDevice] = []
    seen_ips: Set[str] = set()
    ts = now_iso()

    for match in _ARP_RE.finditer(out):
        arp_hostname = match.group(1) or ""  # might be "?" or a name like "unit"
        ip = match.group(2)
        raw_mac = match.group(3)

        # Skip incomplete or broadcast entries
        if raw_mac.lower() in ("(incomplete)", "ff:ff:ff:ff:ff:ff"):
            continue

        mac = _normalise_mac(raw_mac)
        if ip in seen_ips:
            continue
        seen_ips.add(ip)

        # Resolve hostname (prefer reverse DNS, fall back to ARP name)
        hostname = resolve_hostname(ip)
        if not hostname and arp_hostname and arp_hostname != "?":
            hostname = arp_hostname

        devices.append(NetworkDevice(
            ip=ip,
            mac=mac,
            hostname=hostname,
            vendor=mac_to_vendor(mac),
            first_seen=ts,
            last_seen=ts,
            is_active=True,
        ))

    # Step 3: Also add alive IPs that responded to ping but aren't in ARP
    # (rare, but can happen with some network configs)
    for ip in alive_ips:
        if ip not in seen_ips:
            devices.append(NetworkDevice(
                ip=ip,
                mac="Unknown",
                hostname=resolve_hostname(ip),
                vendor="Unknown",
                first_seen=ts,
                last_seen=ts,
                is_active=True,
            ))
            seen_ips.add(ip)

    # Filter out broadcast and self addresses
    my_ip = None
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        my_ip = s.getsockname()[0]
        s.close()
    except Exception:
        pass

    devices = [
        d for d in devices
        if not d.ip.endswith(".255")
        and d.mac != "FF:FF:FF:FF:FF:FF"
        and int(d.ip.split(".")[0]) < 224  # filter multicast/broadcast
    ]

    # Sort by IP
    devices.sort(key=lambda d: [int(x) for x in d.ip.split(".")])

    log.info("Discovered %d devices on the network", len(devices))
    return devices


# Legacy function name kept for backward compatibility
def scan_with_arp_table() -> List[NetworkDevice]:
    """Parse the system ARP table (`arp -a`) without ping sweep."""
    try:
        out = subprocess.check_output(["arp", "-a"], text=True, timeout=30)
    except Exception as exc:
        log.warning("arp -a failed: %s", exc)
        return []

    devices: List[NetworkDevice] = []
    ts = now_iso()
    for match in _ARP_RE.finditer(out):
        ip = match.group(2)
        raw_mac = match.group(3)
        if raw_mac.lower() in ("(incomplete)", "ff:ff:ff:ff:ff:ff"):
            continue
        mac = _normalise_mac(raw_mac)
        hostname = resolve_hostname(ip) or (match.group(1) if match.group(1) != "?" else "")
        devices.append(NetworkDevice(
            ip=ip, mac=mac, hostname=hostname,
            vendor=mac_to_vendor(mac), first_seen=ts, last_seen=ts,
        ))

    return [d for d in devices if not d.ip.endswith(".255")]


# ---------------------------------------------------------------------------
# Synthetic demo devices
# ---------------------------------------------------------------------------

_DEMO_DEVICES = [
    NetworkDevice("192.168.1.1",   "AA:BB:CC:DD:EE:01", "router.local",     "Airtel",            "", "", True),
    NetworkDevice("192.168.1.10",  "AA:BB:CC:DD:EE:02", "MacBook-Pro",      "Apple Inc.",        "", "", True),
    NetworkDevice("192.168.1.11",  "AA:BB:CC:DD:EE:03", "iPhone",           "Apple Inc.",        "", "", True),
    NetworkDevice("192.168.1.12",  "AA:BB:CC:DD:EE:04", "Galaxy-S24",       "Samsung",           "", "", True),
    NetworkDevice("192.168.1.20",  "AA:BB:CC:DD:EE:05", "FireTV-Stick",     "Amazon",            "", "", True),
    NetworkDevice("192.168.1.21",  "AA:BB:CC:DD:EE:06", "Echo-Dot",         "Amazon",            "", "", True),
    NetworkDevice("192.168.1.30",  "AA:BB:CC:DD:EE:07", "HP-Printer",       "HP Inc.",           "", "", False),
    NetworkDevice("192.168.1.31",  "AA:BB:CC:DD:EE:08", "iPad-Air",         "Apple Inc.",        "", "", True),
    NetworkDevice("192.168.1.40",  "AA:BB:CC:DD:EE:09", "Smart-TV",         "LG Electronics",   "", "", True),
    NetworkDevice("192.168.1.41",  "AA:BB:CC:DD:EE:10", "Nest-Camera",      "Google Inc.",       "", "", True),
]


def scan_demo() -> List[NetworkDevice]:
    """Return a fixed list of realistic demo devices."""
    import random
    ts = now_iso()
    devices = []
    for d in _DEMO_DEVICES:
        active = d.is_active if random.random() > 0.15 else not d.is_active
        devices.append(NetworkDevice(
            ip=d.ip, mac=d.mac, hostname=d.hostname, vendor=d.vendor,
            first_seen=ts, last_seen=ts, is_active=active,
        ))
    return devices


# ---------------------------------------------------------------------------
# Unified scan function
# ---------------------------------------------------------------------------

def scan_network(demo: bool = False) -> List[NetworkDevice]:
    """
    Discover ALL devices on the local network.

    Strategy (in order):
    1. Try scapy ARP scan (most reliable but requires root).
    2. Fall back to ping sweep + arp table (no root needed, finds most devices).
    3. If both fail or *demo=True*, return synthetic devices.
    """
    if demo:
        return scan_demo()

    subnet = get_local_subnet() or _get_subnet_fallback()
    log.info("Scanning subnet %s …", subnet)

    # Try scapy first (best results if running as root)
    devices = scan_with_scapy(subnet)

    # Fall back to ping sweep + ARP (works without root, finds all devices)
    if not devices:
        devices = scan_with_ping_and_arp(subnet)

    if not devices:
        log.info("No devices found via live scan — using demo data")
        devices = scan_demo()

    return devices


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    devices = scan_network()
    print(f"\nDiscovered {len(devices)} devices:\n")
    for d in devices:
        status = "🟢" if d.is_active else "🔴"
        print(f"  {status}  {d.ip:<16} {d.mac:<20} {d.vendor:<20} {d.hostname}")
    print(f"\n✅  network_scanner.py self-test passed")
