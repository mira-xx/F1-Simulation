from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np

from .models import TyreCompound, DriverParams, Strategy, Stint
from .simulator import simulate_single_driver  # for base lap model & structure

@dataclass(frozen=True)
class Stochastic:
    pit_mean: float
    pit_std: float
    p_sc: float
    p_vsc: float
    sc_lap_delta: float
    vsc_lap_delta: float
    sc_pit_mult: float
    vsc_pit_mult: float

class MCResult:
    def __init__(self, times_a: List[float], times_b: List[float]):
        self.times_a = np.array(times_a)
        self.times_b = np.array(times_b)

    @property
    def mean_a(self): return float(self.times_a.mean())
    @property
    def p5_a(self):   return float(np.percentile(self.times_a, 5))
    @property
    def p95_a(self):  return float(np.percentile(self.times_a, 95))
    @property
    def mean_b(self): return float(self.times_b.mean())
    @property
    def p5_b(self):   return float(np.percentile(self.times_b, 5))
    @property
    def p95_b(self):  return float(np.percentile(self.times_b, 95))
    @property
    def p_win_a(self): return float((self.times_a < self.times_b).mean())

def simulate_with_randomness(
    race_laps: int,
    strategy: Strategy,
    driver: DriverParams,
    compounds: Dict[str, TyreCompound],
    rng: np.random.Generator,
    stoch: Stochastic,
):
    """
    Single-driver simulation with random SC/VSC and random pit-loss.
    We reuse the deterministic base-lap model from simulate_single_driver,
    but we add per-lap SC/VSC deltas and insert pit losses between stints.
    """
    # Deterministic structure for laps & pit positions
    det = simulate_single_driver(race_laps, strategy, driver, compounds)

    # Copy base lap times and perturb for SC/VSC
    lap_times = []
    neutralized_last_lap = None
    for lap in det.laps:
        # Sample SC/VSC (SC takes precedence)
        neutral = None
        if rng.random() < stoch.p_sc:
            neutral = "SC"
            lap_times.append(lap.lap_time + stoch.sc_lap_delta)
        elif rng.random() < stoch.p_vsc:
            neutral = "VSC"
            lap_times.append(lap.lap_time + stoch.vsc_lap_delta)
        else:
            lap_times.append(lap.lap_time)
        neutralized_last_lap = neutral

    # Insert pit losses after each stint (except last)
    pit_losses = []
    pit_idx_iter = iter(det.pit_laps)  # lap indices after which a pit occurs
    for k, pit_lap in enumerate(det.pit_laps):
        # Use the neutralization state of that lap to scale pit loss
        # (simple heuristic: look at the same-lap neutralization)
        if pit_lap < len(det.laps):
            # Re-sample whether that specific lap was SC/VSC using the same RNG
            # Alternative: track neutralization above in an array.
            # Here we infer from lap_times deltas:
            base = det.laps[pit_lap].lap_time
            delta = lap_times[pit_lap] - base
            if np.isclose(delta, stoch.sc_lap_delta, atol=1e-6):
                mult = stoch.sc_pit_mult
            elif np.isclose(delta, stoch.vsc_lap_delta, atol=1e-6):
                mult = stoch.vsc_pit_mult
            else:
                mult = 1.0
        else:
            mult = 1.0

        pit_loss = rng.normal(loc=stoch.pit_mean * mult, scale=stoch.pit_std)
        pit_losses.append(float(max(0.0, pit_loss)))  # clamp to >= 0

    total_time = float(np.sum(lap_times) + np.sum(pit_losses))
    return total_time

def monte_carlo_pair(
    race_laps: int,
    sa: Strategy,
    sb: Strategy,
    driver_a: DriverParams,
    driver_b: DriverParams,
    compounds: Dict[str, TyreCompound],
    stoch_a: Stochastic,
    stoch_b: Stochastic,
    n_runs: int = 200,
    seed: int = 1,
) -> MCResult:
    rng = np.random.default_rng(seed)
    times_a, times_b = [], []
    for _ in range(n_runs):
        ta = simulate_with_randomness(race_laps, sa, driver_a, compounds, rng, stoch_a)
        tb = simulate_with_randomness(race_laps, sb, driver_b, compounds, rng, stoch_b)
        times_a.append(ta); times_b.append(tb)
    return MCResult(times_a, times_b)
