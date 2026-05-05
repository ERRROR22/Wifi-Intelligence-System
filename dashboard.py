"""
WiFi Intelligence System — Real-Time Monitoring Dashboard
============================================================
Streamlit + Plotly dashboard with a dark "network intelligence center"
aesthetic.  Panels:

1. Network Status       — SSID, channel, RSSI, tx rate
2. Connected Devices    — sortable device table
3. Real-Time Signal     — live RSSI line chart
4. Device Activity      — connection timeline & statistics
5. WiFi Coverage        — interactive heatmap
6. AI Anomaly Alerts    — live ML predictions

Launch:
    streamlit run dashboard.py
"""

from __future__ import annotations

import sys
import os
import datetime
import time

# Ensure the project root is on sys.path so local imports work when
# Streamlit is launched from any working directory.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

# Local modules
from utils import get_logger, now_iso
from wifi_signal_reader import SignalReader
from signal_processing import (
    process_signal_dataframe,
    classify_signal,
)
from network_scanner import scan_network
from device_behavior import DeviceBehaviorTracker, generate_demo_history
from heatmap_mapper import HeatmapMapper, generate_demo_heatmap
from ai_analyzer import AIAnalyzer

log = get_logger("dashboard")

# ---------------------------------------------------------------------------
# Page config & custom CSS
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="WiFi Intelligence System",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
/* Dark intelligence-center theme */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --bg-primary: #0a0e17;
    --bg-secondary: #111827;
    --bg-card: #1a2332;
    --bg-card-hover: #1f2b3d;
    --border-color: #2a3a50;
    --text-primary: #e2e8f0;
    --text-secondary: #94a3b8;
    --accent-blue: #3b82f6;
    --accent-cyan: #06b6d4;
    --accent-green: #22c55e;
    --accent-red: #ef4444;
    --accent-amber: #f59e0b;
    --accent-purple: #a855f7;
    --glow-blue: rgba(59, 130, 246, 0.15);
    --glow-cyan: rgba(6, 182, 212, 0.1);
}

.stApp {
    background: linear-gradient(135deg, var(--bg-primary) 0%, #0d1321 50%, #0a0e17 100%) !important;
    font-family: 'Inter', sans-serif !important;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1525 0%, #111827 100%) !important;
    border-right: 1px solid var(--border-color) !important;
}

section[data-testid="stSidebar"] .stMarkdown h1,
section[data-testid="stSidebar"] .stMarkdown h2,
section[data-testid="stSidebar"] .stMarkdown h3 {
    color: var(--accent-cyan) !important;
    font-weight: 600 !important;
}

/* Metric cards */
div[data-testid="stMetric"] {
    background: linear-gradient(135deg, var(--bg-card) 0%, var(--bg-card-hover) 100%) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 12px !important;
    padding: 16px 20px !important;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3), inset 0 1px 0 rgba(255,255,255,0.03) !important;
    transition: transform 0.2s, box-shadow 0.2s !important;
}

div[data-testid="stMetric"]:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 30px rgba(0, 0, 0, 0.4), 0 0 20px var(--glow-blue) !important;
}

div[data-testid="stMetric"] label {
    color: var(--text-secondary) !important;
    font-weight: 500 !important;
    font-size: 0.8rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.5px !important;
}

div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
    color: var(--accent-cyan) !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-weight: 600 !important;
}

/* Headers */
.stApp h1 {
    color: #f1f5f9 !important;
    font-weight: 700 !important;
    letter-spacing: -0.5px !important;
}

.stApp h2, .stApp h3 {
    color: var(--accent-cyan) !important;
    font-weight: 600 !important;
}

/* Panel containers */
div[data-testid="stExpander"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 12px !important;
    box-shadow: 0 4px 15px rgba(0,0,0,0.25) !important;
}

/* DataFrame styling */
div[data-testid="stDataFrame"] {
    border: 1px solid var(--border-color) !important;
    border-radius: 8px !important;
    overflow: hidden !important;
}

/* Tabs */
button[data-baseweb="tab"] {
    color: var(--text-secondary) !important;
    font-weight: 500 !important;
}

button[data-baseweb="tab"][aria-selected="true"] {
    color: var(--accent-cyan) !important;
    border-bottom-color: var(--accent-cyan) !important;
}

/* Subheader dividers */
.panel-header {
    background: linear-gradient(90deg, var(--accent-cyan), var(--accent-blue));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 1.2rem;
    font-weight: 600;
    padding-bottom: 4px;
    border-bottom: 2px solid var(--border-color);
    margin-bottom: 12px;
}

/* Glowing status indicators */
.status-online  { color: var(--accent-green); text-shadow: 0 0 8px rgba(34,197,94,0.5); }
.status-offline { color: var(--accent-red); text-shadow: 0 0 8px rgba(239,68,68,0.4); }
.status-warn    { color: var(--accent-amber); text-shadow: 0 0 8px rgba(245,158,11,0.4); }

/* Scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg-primary); }
::-webkit-scrollbar-thumb { background: var(--border-color); border-radius: 3px; }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Session-state initialisation
# ---------------------------------------------------------------------------

def _init_state():
    """Initialise long-lived objects in Streamlit session state."""
    if "signal_reader" not in st.session_state:
        # Use 3s interval since system_profiler is slower than airport
        st.session_state.signal_reader = SignalReader(max_history=600, interval=3.0)
        st.session_state.signal_reader.start()

    if "device_tracker" not in st.session_state:
        reader_ref = st.session_state.signal_reader
        if reader_ref.is_demo:
            # Demo mode: pre-load synthetic history
            st.session_state.device_tracker = generate_demo_history(hours=12)
        else:
            # Live mode: start fresh tracker for real device data
            st.session_state.device_tracker = DeviceBehaviorTracker()

    if "heatmap_mapper" not in st.session_state:
        reader_ref = st.session_state.signal_reader
        if reader_ref.is_demo:
            st.session_state.heatmap_mapper = generate_demo_heatmap()
        else:
            # Live mode: start with empty mapper for real measurements
            st.session_state.heatmap_mapper = HeatmapMapper()

    if "ai_analyzer" not in st.session_state:
        analyzer = AIAnalyzer()
        if not analyzer.load_models():
            analyzer.train()
        st.session_state.ai_analyzer = analyzer

    if "last_scan_devices" not in st.session_state:
        # Always try real network scan first
        st.session_state.last_scan_devices = scan_network(demo=False)

    if "auto_refresh" not in st.session_state:
        st.session_state.auto_refresh = True


_init_state()

reader: SignalReader = st.session_state.signal_reader
tracker: DeviceBehaviorTracker = st.session_state.device_tracker
mapper: HeatmapMapper = st.session_state.heatmap_mapper
analyzer: AIAnalyzer = st.session_state.ai_analyzer

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("# 📡 WiFi Intel")
    st.markdown("**Network Intelligence Center**")
    st.divider()

    mode_label = "🟡 DEMO" if reader.is_demo else "🟢 LIVE"
    st.markdown(f"### Mode: {mode_label}")

    st.session_state.auto_refresh = st.toggle("Auto-Refresh (5s)", value=st.session_state.auto_refresh)

    st.divider()
    st.markdown("### 🔍 Actions")

    if st.button("🔄 Scan Network", use_container_width=True):
        with st.spinner("Scanning …"):
            devices = scan_network(demo=False)  # Always try real scan
            st.session_state.last_scan_devices = devices
            tracker.observe(devices)
            tracker.save_history()
            st.success(f"Found {len(devices)} devices")

    if st.button("🧠 Retrain AI Models", use_container_width=True):
        with st.spinner("Training …"):
            analyzer.train()
            analyzer.save_models()
            st.success("Models retrained!")

    st.divider()
    st.markdown("### 📐 Heatmap Measurement")
    st.caption("Walk to a spot in your room, enter coordinates, and capture the live RSSI.")
    with st.form("heatmap_form"):
        hm_x = st.number_input("X position (meters)", 0.0, mapper.room_width, 5.0, 0.5)
        hm_y = st.number_input("Y position (meters)", 0.0, mapper.room_height, 4.0, 0.5)
        latest = reader.latest
        current_rssi = latest.rssi if latest else -55.0
        st.markdown(f"**📡 Current RSSI: `{current_rssi} dBm`**")
        hm_rssi = st.number_input(
            "RSSI (dBm) — auto-filled from live signal",
            -100.0, 0.0,
            current_rssi,
            1.0,
        )
        submitted = st.form_submit_button("📍 Capture Point")
        if submitted:
            mapper.add_measurement(hm_x, hm_y, hm_rssi)
            mapper.save()
            st.success(f"✅ Point ({hm_x}, {hm_y}) @ {hm_rssi} dBm added!  ({len(mapper.measurements)} total)")

    if mapper.measurements:
        st.caption(f"📊 {len(mapper.measurements)} points collected")
        if st.button("🗑️ Clear All Points", use_container_width=True):
            mapper.clear()
            mapper.save()
            st.success("Heatmap cleared!")

    st.divider()
    st.caption(f"Last update: {now_iso()}")

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.markdown(
    """
    <h1 style='text-align:center; margin-bottom:0;'>
        📡 WiFi Intelligence System
    </h1>
    <p style='text-align:center; color:#94a3b8; margin-top:4px; font-size:1rem;'>
        Real-Time Network Monitoring · Device Analytics · AI Anomaly Detection
    </p>
    """,
    unsafe_allow_html=True,
)

st.markdown("---")

# ---------------------------------------------------------------------------
# Panel 1: Network Status (top metrics row)
# ---------------------------------------------------------------------------

latest_sample = reader.latest

col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    ssid = latest_sample.ssid if latest_sample else "—"
    st.metric("SSID", ssid)
with col2:
    rssi_val = latest_sample.rssi if latest_sample else 0
    quality = classify_signal(rssi_val) if latest_sample else "—"
    st.metric("Signal (RSSI)", f"{rssi_val} dBm", delta=quality)
with col3:
    noise_val = latest_sample.noise if latest_sample else 0
    st.metric("Noise Floor", f"{noise_val} dBm")
with col4:
    chan = latest_sample.channel if latest_sample else "—"
    st.metric("Channel", chan)
with col5:
    tx = latest_sample.tx_rate if latest_sample else 0
    st.metric("TX Rate", f"{tx} Mbps")

st.markdown("")

# ---------------------------------------------------------------------------
# Panel 2 & 3 side-by-side: Devices table + Real-Time Signal
# ---------------------------------------------------------------------------

left_col, right_col = st.columns([2, 3])

# -- Connected Devices -----------------------------------------------------
with left_col:
    st.markdown("### 🖥️ Connected Devices")
    devices = st.session_state.last_scan_devices
    if devices:
        dev_df = pd.DataFrame([d.to_dict() for d in devices])
        display_cols = ["ip", "mac", "hostname", "vendor", "is_active"]
        dev_df["is_active"] = dev_df["is_active"].map({True: "🟢 Active", False: "🔴 Inactive"})
        st.dataframe(
            dev_df[display_cols].rename(columns={
                "ip": "IP Address",
                "mac": "MAC Address",
                "hostname": "Hostname",
                "vendor": "Vendor",
                "is_active": "Status",
            }),
            use_container_width=True,
            height=380,
        )
    else:
        st.info("No devices discovered yet. Click **Scan Network** in the sidebar.")

# -- Real-Time Signal Graph ------------------------------------------------
with right_col:
    st.markdown("### 📈 Real-Time RSSI Signal")
    history = reader.history_dicts()
    if history:
        sig_df = pd.DataFrame(history)
        sig_df["timestamp"] = pd.to_datetime(sig_df["timestamp"])
        processed = process_signal_dataframe(sig_df)

        fig_signal = go.Figure()
        fig_signal.add_trace(go.Scatter(
            x=processed["timestamp"], y=processed["rssi"],
            mode="lines",
            name="Raw RSSI",
            line=dict(color="#3b82f6", width=1),
            opacity=0.5,
        ))
        fig_signal.add_trace(go.Scatter(
            x=processed["timestamp"], y=processed["rssi_ema"],
            mode="lines",
            name="EMA (smoothed)",
            line=dict(color="#06b6d4", width=2.5),
        ))
        # Anomaly markers
        anomalies = processed[processed["is_anomaly"]]
        if not anomalies.empty:
            fig_signal.add_trace(go.Scatter(
                x=anomalies["timestamp"], y=anomalies["rssi"],
                mode="markers",
                name="Anomaly",
                marker=dict(color="#ef4444", size=9, symbol="x"),
            ))

        fig_signal.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(26,35,50,0.6)",
            height=380,
            margin=dict(l=20, r=20, t=30, b=20),
            xaxis_title="Time",
            yaxis_title="RSSI (dBm)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            yaxis=dict(range=[-90, -25]),
        )
        st.plotly_chart(fig_signal, use_container_width=True)
    else:
        st.info("Collecting signal data …")

# ---------------------------------------------------------------------------
# Panel 4 & 5: Device Activity + Coverage Heatmap
# ---------------------------------------------------------------------------

st.markdown("---")
bottom_left, bottom_right = st.columns(2)

# -- Device Activity -------------------------------------------------------
with bottom_left:
    st.markdown("### 📊 Device Activity")

    tab_summary, tab_freq, tab_timeline = st.tabs(["Summary", "Connection Frequency", "Timeline"])

    with tab_summary:
        summary = tracker.summary_dict()
        s1, s2, s3 = st.columns(3)
        with s1:
            st.metric("Total Devices", summary["total_devices_seen"])
        with s2:
            st.metric("Active Now", summary["currently_active"])
        with s3:
            st.metric("Peak Usage", summary["peak_usage_time"])

        st.markdown(f"**Most Active:** {summary['most_active_device']} ({summary['most_active_vendor']})")
        st.markdown(f"**Total Scans:** {summary['total_scans']}")

    with tab_freq:
        freq_df = tracker.connection_frequency()
        if not freq_df.empty:
            fig_freq = px.bar(
                freq_df.head(10),
                x="hostname",
                y="active_count",
                color="activity_ratio",
                color_continuous_scale="Viridis",
                labels={"hostname": "Device", "active_count": "Active Count", "activity_ratio": "Activity %"},
            )
            fig_freq.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(26,35,50,0.6)",
                height=320,
                margin=dict(l=20, r=20, t=30, b=20),
            )
            st.plotly_chart(fig_freq, use_container_width=True)
        else:
            st.info("No activity data yet.")

    with tab_timeline:
        tl_df = tracker.scan_timeline()
        if not tl_df.empty:
            tl_df["timestamp"] = pd.to_datetime(tl_df["timestamp"])
            fig_tl = go.Figure()
            fig_tl.add_trace(go.Scatter(
                x=tl_df["timestamp"], y=tl_df["device_count"],
                fill="tozeroy",
                name="Total Devices",
                line=dict(color="#a855f7"),
                fillcolor="rgba(168,85,247,0.15)",
            ))
            fig_tl.add_trace(go.Scatter(
                x=tl_df["timestamp"], y=tl_df["active_count"],
                fill="tozeroy",
                name="Active Devices",
                line=dict(color="#22c55e"),
                fillcolor="rgba(34,197,94,0.12)",
            ))
            fig_tl.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(26,35,50,0.6)",
                height=320,
                margin=dict(l=20, r=20, t=30, b=20),
                xaxis_title="Time",
                yaxis_title="Count",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            st.plotly_chart(fig_tl, use_container_width=True)
        else:
            st.info("No timeline data yet.")

# -- WiFi Coverage Heatmap -------------------------------------------------
with bottom_right:
    st.markdown("### 🗺️ WiFi Coverage Heatmap")
    if len(mapper.measurements) >= 3:
        try:
            heatmap_fig = mapper.generate_plotly_heatmap()
            heatmap_fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                height=420,
                margin=dict(l=20, r=20, t=40, b=20),
            )
            st.plotly_chart(heatmap_fig, use_container_width=True)
        except Exception as e:
            st.error(f"Heatmap error: {e}")
    else:
        st.info("Add at least 3 measurement points from the sidebar to generate a heatmap.")

# ---------------------------------------------------------------------------
# Panel 6: AI Anomaly Detection
# ---------------------------------------------------------------------------

st.markdown("---")
st.markdown("### 🤖 AI Anomaly Detection")

ai_col1, ai_col2 = st.columns([3, 2])

with ai_col1:
    # Run prediction on current data
    if history:
        recent = history[-min(30, len(history)):]
        rssi_vals = [s["rssi"] for s in recent]
        noise_vals = [s["noise"] for s in recent]
        device_count = len(devices) if devices else 5

        features = AIAnalyzer.extract_features(
            rssi_values=rssi_vals,
            noise_values=noise_vals,
            device_count=device_count,
        )

        rf_pred = analyzer.predict(features, model="random_forest")[0]
        lr_pred = analyzer.predict(features, model="logistic_regression")[0]
        rf_proba = analyzer.predict_proba(features, model="random_forest")[0]
        lr_proba = analyzer.predict_proba(features, model="logistic_regression")[0]

        p1, p2 = st.columns(2)
        with p1:
            label = "⚠️ ANOMALY DETECTED" if rf_pred == 1 else "✅ Normal"
            color = "🔴" if rf_pred == 1 else "🟢"
            st.metric(
                "Random Forest",
                label,
                delta=f"Confidence: {max(rf_proba):.0%}",
            )
        with p2:
            label = "⚠️ ANOMALY DETECTED" if lr_pred == 1 else "✅ Normal"
            st.metric(
                "Logistic Regression",
                label,
                delta=f"Confidence: {max(lr_proba):.0%}",
            )

        # Feature values
        with st.expander("📋 Current Feature Vector"):
            feat_df = pd.DataFrame([features])
            st.dataframe(feat_df, use_container_width=True)
    else:
        st.info("Waiting for signal data to run predictions …")

with ai_col2:
    st.markdown("**Model Performance**")
    metrics = analyzer.metrics
    if metrics:
        for name, m in metrics.items():
            st.markdown(f"**{name.replace('_', ' ').title()}**")
            st.markdown(f"- Accuracy: `{m['accuracy']:.2%}`")
            if "1" in m.get("report", {}):
                prec = m["report"]["1"].get("precision", 0)
                rec = m["report"]["1"].get("recall", 0)
                st.markdown(f"- Anomaly Precision: `{prec:.2%}`")
                st.markdown(f"- Anomaly Recall: `{rec:.2%}`")
    else:
        st.info("Models not yet trained.")

    # Feature importance chart
    imp_df = analyzer.feature_importances()
    if not imp_df.empty:
        fig_imp = px.bar(
            imp_df,
            x="importance",
            y="feature",
            orientation="h",
            color="importance",
            color_continuous_scale="Cividis",
        )
        fig_imp.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(26,35,50,0.6)",
            height=280,
            margin=dict(l=10, r=10, t=10, b=10),
            showlegend=False,
            coloraxis_showscale=False,
            yaxis=dict(autorange="reversed"),
        )
        st.plotly_chart(fig_imp, use_container_width=True)

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.markdown("---")
st.markdown(
    """
    <div style='text-align:center; color:#64748b; font-size:0.8rem; padding:8px;'>
        WiFi Intelligence System · Built with Python, Streamlit & Plotly ·
        AI-Powered Network Monitoring
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Auto-refresh
# ---------------------------------------------------------------------------

if st.session_state.auto_refresh:
    time.sleep(5)
    st.rerun()
