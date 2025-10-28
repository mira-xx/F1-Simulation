from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass

@dataclass
class Track:
    s: np.ndarray          # arc-length [m], shape (N,)
    x: np.ndarray          # [m]
    y: np.ndarray          # [m]
    kappa: np.ndarray      # curvature κ(s) [1/m]
    ds: float              # step [m]
    drs_mask: np.ndarray   # bool mask for DRS zones (optional)

def _central_diff(arr: np.ndarray, ds: float) -> np.ndarray:
    d = np.zeros_like(arr)
    d[1:-1] = (arr[2:] - arr[:-2]) / (2*ds)
    d[0]    = (arr[1] - arr[0]) / ds
    d[-1]   = (arr[-1] - arr[-2]) / ds
    return d

def resample_polyline(x: np.ndarray, y: np.ndarray, ds_target: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Uniformly resample a closed/open polyline at ~ds_target spacing."""
    seg = np.hypot(np.diff(x), np.diff(y))
    s_raw = np.concatenate(([0.0], np.cumsum(seg)))
    s_total = s_raw[-1]
    N = max(10, int(np.round(s_total / ds_target)))
    s = np.linspace(0.0, s_total, N)
    x_u = np.interp(s, s_raw, x)
    y_u = np.interp(s, s_raw, y)
    return s, x_u, y_u

def curvature_from_xy(x: np.ndarray, y: np.ndarray, s: np.ndarray) -> np.ndarray:
    ds = np.gradient(s)
    dx = _central_diff(x, ds.mean())
    dy = _central_diff(y, ds.mean())
    ddx = _central_diff(dx, ds.mean())
    ddy = _central_diff(dy, ds.mean())
    num = dx*ddy - dy*ddx
    den = (dx*dx + dy*dy)**1.5 + 1e-12
    kappa = num / den
    return kappa

def load_track_csv(path: str, ds: float = 2.0, drs_ranges_m: list[tuple[float,float]] | None = None) -> Track:
    """
    CSV must have columns: x,y  (meters). Row order should follow the lap path.
    ds: target resampling step in meters.
    drs_ranges_m: list of (s_start, s_end) for DRS zones, in meters along lap.
    """
    df = pd.read_csv(path)
    x, y = df["x"].values.astype(float), df["y"].values.astype(float)
    s, x_u, y_u = resample_polyline(x, y, ds)
    kappa = curvature_from_xy(x_u, y_u, s)
    drs_mask = np.zeros_like(s, dtype=bool)
    if drs_ranges_m:
        for a, b in drs_ranges_m:
            drs_mask |= (s >= a) & (s <= b)
    return Track(s=s, x=x_u, y=y_u, kappa=kappa, ds=float(s[1]-s[0]), drs_mask=drs_mask)
