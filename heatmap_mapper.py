"""
WiFi Intelligence System — Heatmap Mapper
============================================
Collects WiFi signal measurements at user-specified (x, y) coordinates and
generates interpolated 2-D heatmaps of WiFi coverage.

Colour scheme:
  🟢 green  → strong signal (≥ -50 dBm)
  🟡 yellow → medium signal
  🔴 red    → weak signal   (≤ -80 dBm)
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import plotly.graph_objects as go
from scipy.interpolate import griddata

from utils import DATASET_DIR, get_logger

log = get_logger("heatmap")

HEATMAP_FILE = DATASET_DIR / "heatmap_data.json"

# ---------------------------------------------------------------------------
# Measurement model
# ---------------------------------------------------------------------------

class HeatmapMapper:
    """Collect (x, y, rssi) measurements and produce interpolated heatmaps."""

    def __init__(self, room_width: float = 10.0, room_height: float = 8.0):
        self.room_width = room_width
        self.room_height = room_height
        self.measurements: List[Dict] = []

    # -- data collection ---------------------------------------------------

    def add_measurement(self, x: float, y: float, rssi: float):
        """Record a single measurement point."""
        self.measurements.append({"x": x, "y": y, "rssi": rssi})
        log.debug("Added point (%.1f, %.1f) RSSI=%.1f", x, y, rssi)

    def clear(self):
        self.measurements.clear()

    # -- persistence -------------------------------------------------------

    def save(self, path: Optional[Path] = None):
        path = path or HEATMAP_FILE
        data = {
            "room_width": self.room_width,
            "room_height": self.room_height,
            "measurements": self.measurements,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        log.info("Saved %d measurements to %s", len(self.measurements), path)

    def load(self, path: Optional[Path] = None):
        path = path or HEATMAP_FILE
        with open(path) as f:
            data = json.load(f)
        self.room_width = data.get("room_width", self.room_width)
        self.room_height = data.get("room_height", self.room_height)
        self.measurements = data.get("measurements", [])
        log.info("Loaded %d measurements from %s", len(self.measurements), path)

    # -- interpolation & heatmap ------------------------------------------

    def interpolate(self, resolution: int = 100) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Interpolate the sparse measurement points onto a dense grid.

        Returns (X, Y, Z) meshgrid arrays suitable for plotting.
        """
        if len(self.measurements) < 3:
            raise ValueError("Need at least 3 measurement points to interpolate")

        xs = np.array([m["x"] for m in self.measurements])
        ys = np.array([m["y"] for m in self.measurements])
        zs = np.array([m["rssi"] for m in self.measurements])

        grid_x = np.linspace(0, self.room_width, resolution)
        grid_y = np.linspace(0, self.room_height, resolution)
        X, Y = np.meshgrid(grid_x, grid_y)

        Z = griddata((xs, ys), zs, (X, Y), method="cubic", fill_value=np.nanmin(zs))

        return X, Y, Z

    def generate_plotly_heatmap(self, resolution: int = 100) -> go.Figure:
        """
        Return a Plotly Figure with the interpolated WiFi heatmap.
        Uses a green → yellow → red colour gradient.
        """
        X, Y, Z = self.interpolate(resolution)

        # Custom colour scale: strong (green) → weak (red)
        colorscale = [
            [0.0, "rgb(220,  50,  50)"],   # very weak  (most negative dBm)
            [0.3, "rgb(240, 150,  30)"],   # weak
            [0.5, "rgb(250, 230,  50)"],   # medium
            [0.7, "rgb(140, 220,  60)"],   # good
            [1.0, "rgb( 30, 180,  60)"],   # strong     (least negative dBm)
        ]

        fig = go.Figure(data=go.Heatmap(
            x=np.linspace(0, self.room_width, resolution),
            y=np.linspace(0, self.room_height, resolution),
            z=Z,
            colorscale=colorscale,
            colorbar=dict(title="RSSI (dBm)", ticksuffix=" dBm"),
            hovertemplate="X: %{x:.1f}m<br>Y: %{y:.1f}m<br>RSSI: %{z:.1f} dBm<extra></extra>",
        ))

        # Overlay measurement points
        mx = [m["x"] for m in self.measurements]
        my = [m["y"] for m in self.measurements]
        mz = [m["rssi"] for m in self.measurements]
        fig.add_trace(go.Scatter(
            x=mx, y=my,
            mode="markers+text",
            marker=dict(size=10, color="white", line=dict(width=2, color="black")),
            text=[f"{z:.0f}" for z in mz],
            textposition="top center",
            textfont=dict(color="white", size=10),
            name="Measurements",
            hovertemplate="X: %{x:.1f}m<br>Y: %{y:.1f}m<br>RSSI: %{text} dBm<extra></extra>",
        ))

        fig.update_layout(
            title="WiFi Signal Coverage Heatmap",
            xaxis_title="Room Width (m)",
            yaxis_title="Room Height (m)",
            template="plotly_dark",
            width=700,
            height=550,
            xaxis=dict(range=[0, self.room_width], constrain="domain"),
            yaxis=dict(range=[0, self.room_height], scaleanchor="x"),
        )
        return fig


# ---------------------------------------------------------------------------
# Demo data generator
# ---------------------------------------------------------------------------

def generate_demo_heatmap(
    room_w: float = 10.0,
    room_h: float = 8.0,
    router_x: float = 5.0,
    router_y: float = 0.5,
    num_points: int = 30,
) -> HeatmapMapper:
    """
    Generate realistic demo measurements simulating signal attenuation
    with distance from a router placed at (*router_x*, *router_y*).
    """
    mapper = HeatmapMapper(room_width=room_w, room_height=room_h)

    for _ in range(num_points):
        x = round(random.uniform(0.5, room_w - 0.5), 1)
        y = round(random.uniform(0.5, room_h - 0.5), 1)
        dist = math.sqrt((x - router_x) ** 2 + (y - router_y) ** 2)
        # Free-space path loss approximation + random wall attenuation
        rssi = -30 - 20 * math.log10(max(dist, 0.3)) - random.uniform(0, 8)
        rssi = round(max(rssi, -90), 1)
        mapper.add_measurement(x, y, rssi)

    # Add the router location itself
    mapper.add_measurement(router_x, router_y, round(-28 + random.gauss(0, 1), 1))

    return mapper


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    mapper = generate_demo_heatmap()
    print(f"Generated {len(mapper.measurements)} measurement points")
    mapper.save()
    fig = mapper.generate_plotly_heatmap()
    fig.write_html(str(DATASET_DIR / "heatmap_preview.html"))
    print(f"Preview saved to {DATASET_DIR / 'heatmap_preview.html'}")
    print("✅  heatmap_mapper.py self-test passed")
