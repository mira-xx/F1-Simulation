from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from .track_tools import Track

g = 9.81

@dataclass
class CarParams:
    mass: float         # kg
    CdA: float          # drag area (Cd*A) [m^2]
    ClA: float          # downforce area (Cl*A) [m^2]
    mu: float           # tyre-road friction coefficient
    P_max: float        # max engine power [W]
    F_brake_max: float  # max braking force at tyre limit [N]
    rho: float = 1.2    # air density [kg/m^3]

@dataclass
class SolveOut:
    v: np.ndarray        # speed profile [m/s]
    a_long: np.ndarray   # longitudinal accel [m/s^2]
    lap_time: float      # [s]

def aero_forces(v: np.ndarray, params: CarParams) -> tuple[np.ndarray, np.ndarray]:
    q = 0.5 * params.rho * v*v
    F_drag = q * params.CdA
    F_down = q * params.ClA
    return F_drag, F_down

def corner_speed_cap(track: Track, params: CarParams) -> np.ndarray:
    # v^2 / R <= mu * (g + F_down/m)  ->  v^2 <= mu * R * (g + (0.5*rho*ClA*v^2)/m)
    # Solve: v_cap from quadratic: v^2 - mu*R*(0.5*rho*ClA/m)*v^2 <= mu*R*g
    # => v^2 * (1 - mu*R*(0.5*rho*ClA/m)) <= mu*R*g
    # For numerical stability, iterate once:
    R = np.where(np.abs(track.kappa) > 1e-6, 1.0/np.abs(track.kappa), 1e9)
    v = np.sqrt(np.maximum(0.0, params.mu * R * g))  # no aero first
    for _ in range(2):
        _, F_down = aero_forces(v, params)
        a_lat_max = params.mu * (g + F_down/params.mass)
        v = np.sqrt(np.maximum(0.0, a_lat_max * R))
    return np.clip(v, 0.1, 150.0)  # 540 km/h cap for safety

def forward_backward_profile(track: Track, params: CarParams, v_corner_cap: np.ndarray) -> SolveOut:
    N, ds = len(track.s), track.ds
    v_f = np.minimum(v_corner_cap.copy(), np.full(N, 150.0))

    # ---------- Forward pass (accel-limited) ----------
    v = v_f
    for i in range(1, N):
        v_prev = v[i-1]
        # Available tractive force with friction circle vs aero/drag
        F_drag, F_down = aero_forces(np.array([v_prev]), params)
        a_lat = (v_prev**2) * np.abs(track.kappa[i-1])
        a_max = params.mu * (g + F_down[0]/params.mass)
        a_long_cap = np.sqrt(max(0.0, a_max*a_max - a_lat*a_lat))  # friction circle
        # Power limit: P = F*v -> F = P/v (avoid div by zero)
        F_power = params.P_max / max(v_prev, 1e-3)
        F_tractive = min(F_power, a_long_cap*params.mass)
        # Update with constant-accel over ds: v^2 = v0^2 + 2*a*ds  ->  a_net = (F_tractive - F_drag)/m
        a_net = (F_tractive - F_drag[0]) / params.mass
        v[i] = min(v[i], np.sqrt(max(0.0, v_prev*v_prev + 2*a_net*ds)))

    # ---------- Backward pass (brake-limited) ----------
    for i in range(N-2, -1, -1):
        v_next = v[i+1]
        F_drag, F_down = aero_forces(np.array([v_next]), params)
        a_lat = (v_next**2) * np.abs(track.kappa[i+1])
        a_max = params.mu * (g + F_down[0]/params.mass)
        a_long_cap = np.sqrt(max(0.0, a_max*a_max - a_lat*a_lat))
        # Braking limited by tyre (friction circle) and hardware cap
        a_brake_tyre = a_long_cap
        a_brake_hardware = params.F_brake_max / params.mass
        a_brake = min(a_brake_tyre, a_brake_hardware)
        # Integrate backwards: v0^2 = v1^2 + 2*a*ds  (a is negative)
        a_net = -(a_brake + F_drag[0]/params.mass)
        v_allowed = np.sqrt(max(0.0, v_next*v_next + 2*a_net*ds))
        v[i] = min(v[i], v_allowed)

    # Longitudinal accel estimate (finite diff in s, then convert)
    a_long = np.zeros_like(v)
    a_long[1:] = (v[1:]**2 - v[:-1]**2) / (2*ds)
    # Time integration
    dt = 2*ds / (v[1:] + v[:-1] + 1e-9)
    lap_time = float(np.sum(dt))
    return SolveOut(v=v, a_long=a_long, lap_time=lap_time)
