# 📡 WiFi Intelligence System

> **AI-Powered WiFi Network Analysis & Signal Heatmap Mapper**

A complete network intelligence platform built in Python for macOS. It discovers devices, monitors WiFi signal strength, generates coverage heatmaps, and performs ML-based anomaly detection — all visualised through a real-time Streamlit dashboard.

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd wifi_intelligence_system
pip install -r requirements.txt
```

### 2. Launch the Dashboard

```bash
streamlit run dashboard.py
```

The dashboard starts in **demo mode** automatically if you're not connected to WiFi or not on macOS. When running on a MacBook with WiFi, it captures live RSSI data using the system `airport` utility.

> **Note:** Network scanning via scapy requires `sudo` for ARP scanning. Without root privileges the system falls back to parsing the system ARP table (`arp -a`).

---

## 📂 Project Structure

```
wifi_intelligence_system/
├── dashboard.py            # Streamlit real-time dashboard
├── utils.py                # Shared utilities (logging, MAC lookup, paths)
├── wifi_signal_reader.py   # RSSI capture via macOS airport
├── signal_processing.py    # Smoothing, filtering, anomaly flagging
├── network_scanner.py      # Device discovery (ARP scan / arp table)
├── device_behavior.py      # Device activity tracking & analytics
├── heatmap_mapper.py       # Coordinate-based WiFi coverage mapping
├── ai_analyzer.py          # ML anomaly detection (RF + LR)
├── requirements.txt        # pip dependencies
├── README.md               # ← you are here
└── dataset/
    ├── example_signal_data.csv
    ├── example_device_history.csv
    └── example_heatmap_data.json
```

---

## 🧩 System Modules

| Module | Purpose |
|--------|---------|
| `wifi_signal_reader.py` | Calls `/System/Library/PrivateFrameworks/Apple80211.framework/…/airport -I` to read RSSI, noise, SSID, channel, and TX rate at ~1 Hz |
| `signal_processing.py` | Moving average, EMA smoothing, rolling variance, z-score anomaly detection |
| `network_scanner.py` | ARP scan (scapy) or `arp -a` fallback to discover IP, MAC, hostname, vendor |
| `device_behavior.py` | Tracks device connection history, peak usage, most-active device |
| `heatmap_mapper.py` | Collects (x, y, RSSI) points, interpolates with scipy, renders Plotly heatmap |
| `ai_analyzer.py` | Random Forest & Logistic Regression for network anomaly classification |
| `dashboard.py` | Streamlit + Plotly real-time UI with 6 intelligence panels |

---

## 📊 Dashboard Panels

1. **Network Status** — SSID, RSSI, noise floor, channel, TX rate
2. **Connected Devices** — Live table with vendor info and active/inactive status
3. **Real-Time Signal** — RSSI line chart with EMA smoothing and anomaly markers
4. **Device Activity** — Summary stats, connection frequency bars, activity timeline
5. **WiFi Coverage** — Interpolated heatmap (green=strong → red=weak)
6. **AI Anomaly Detection** — Live Random Forest / Logistic Regression predictions with feature importance

---

## 📡 How WiFi RSSI Works

**RSSI** (Received Signal Strength Indicator) measures the power of a received WiFi radio signal in **dBm** (decibels relative to 1 milliwatt).

| RSSI Range | Quality |
|------------|---------|
| ≥ −50 dBm | Excellent |
| −50 to −60 | Good |
| −60 to −70 | Fair |
| −70 to −80 | Weak |
| ≤ −80 dBm | Very Weak |

The macOS `airport -I` command provides:
- **agrCtlRSSI** — current signal strength
- **agrCtlNoise** — ambient noise floor
- **SNR** (Signal-to-Noise Ratio) = RSSI − Noise

### How Network Scanning Works

The system sends ARP (Address Resolution Protocol) broadcast packets to every IP on the local subnet. Devices that respond reveal their **MAC address** and **IP address**. The MAC's first 3 octets (OUI) identify the manufacturer, enabling vendor lookup.

### Limitations of RSSI-Based Sensing

| Limitation | Details |
|------------|---------|
| **Low resolution** | RSSI is a single scalar — it cannot distinguish multipath reflections |
| **Environmental noise** | Walls, furniture, people, and appliances cause unpredictable attenuation |
| **Device variation** | Different hardware reports different RSSI for the same signal |
| **No direction** | RSSI gives magnitude only, not angle of arrival |
| **Slow updates** | Typical sampling is 1–10 Hz, too slow for fine-grained motion detection |

### Potential Improvements with CSI Data

**CSI** (Channel State Information) provides per-subcarrier amplitude and phase data from WiFi OFDM signals. Compared to RSSI:

- **30–256 subcarriers** instead of 1 value → much richer feature space
- **Phase information** enables angle-of-arrival estimation
- **Sub-wavelength sensitivity** can detect breathing and hand gestures
- **Requires** specialised hardware (Intel 5300, Atheros 9580, ESP32-S3) or tools like [Nexmon CSI](https://github.com/seemoo-lab/nexmon_csi)

CSI would dramatically improve the ML models in `ai_analyzer.py` for applications such as human activity recognition, room occupancy counting, and gesture detection.

---

## 📁 Example Dataset Format

### `example_signal_data.csv`
```
timestamp,rssi,noise,ssid,channel,tx_rate
2026-03-16T20:00:00,-55.2,-95.1,Airtel_Xstream,36,866
2026-03-16T20:00:01,-54.8,-94.9,Airtel_Xstream,36,780
```

### `example_device_history.csv`
```
mac,ip,hostname,vendor,first_seen,last_seen,seen_count,active_count,inactive_count,total_minutes_online,activity_ratio
AA:BB:CC:DD:EE:01,192.168.1.1,router.local,Airtel,2026-03-16T08:00:00,2026-03-16T20:00:00,48,46,2,46.0,0.96
```

### `example_heatmap_data.json`
```json
{
  "room_width": 10.0,
  "room_height": 8.0,
  "measurements": [
    {"x": 5.0, "y": 0.5, "rssi": -28.3},
    {"x": 2.0, "y": 4.0, "rssi": -52.7},
    {"x": 8.0, "y": 7.0, "rssi": -68.1}
  ]
}
```

---

## 🛠️ Running Individual Modules

Each module has a self-test block you can run independently:

```bash
python utils.py
python wifi_signal_reader.py
python signal_processing.py
python network_scanner.py
python device_behavior.py
python heatmap_mapper.py
python ai_analyzer.py
```

---

## 📜 License

This project is for educational and portfolio purposes. Built with ❤️ using Python, Streamlit, Plotly, scikit-learn, scapy, and scipy.


    ##Ritik Sharma