from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np

from .models import TyreCompound, DriverParams, Strategy, Stint
from .simulator import simulate_single_driver  # deterministic base

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
        self.times_a = np.array(times_a, dtype=float)
        self.times_b = np.array(times_b, dtype=float)
    @property
    def mean_a(self) -> float: return float(self.times_a.mean())
    @property
    def p5_a(self)   -> float: return float(np.percentile(self.times_a, 5))
    @property
    def p95_a(self)  -> float: return float(np.percentile(self.times_a, 95))
    @property
    def mean_b(self) -> float: return float(self.times_b.mean())
    @property
    def p5_b(self)   -> float: return float(np.percentile(self.times_b, 5))
    @property
    def p95_b(self)  -> float: return float(np.percentile(self.times_b, 95))
    @property
    def p_win_a(self)-> float: return float((self.times_a < self.times_b).mean())

# ---------- Memoization of deterministic plan ----------
# Build a stable key for caching the deterministic structure
def _compounds_hash(compounds: Dict[str, TyreCompound]) -> Tuple:
    # order by name so dict order doesn't matter
    return tuple(sorted((k, v.deg_rate, v.base_offset) for k, v in compounds.items()))

def _strategy_key(strategy: Strategy) -> Tuple[Tuple[str, int], ...]:
    return tuple((s.compound, s.laps) for s in strategy.stints)

def _driver_key(driver: DriverParams) -> Tuple[float, float]:
    return (driver.base_pace, driver.fuel_penalty)

# Cache: maps (race_laps, driver_key, strategy_key, compounds_hash) -> (base_lap_times: np.ndarray, pit_laps: List[int])
_PLAN_CACHE: Dict[Tuple, Tuple[np.ndarray, List[int]]] = {}

def _get_plan(
    race_laps: int,
    strategy: Strategy,
    driver: DriverParams,
    compounds: Dict[str, TyreCompound],
) -> Tuple[np.ndarray, List[int]]:
    key = (race_laps, _driver_key(driver), _strategy_key(strategy), _compounds_hash(compounds))
    cached = _PLAN_CACHE.get(key)
    if cached is not None:
        return cached
    # Build once via deterministic simulator
    det = simulate_single_driver(race_laps, strategy, driver, compounds)
    base = np.array([lap.lap_time for lap in det.laps], dtype=float)
    _PLAN_CACHE[key] = (base, det.pit_laps)
    return _PLAN_CACHE[key]

# ---------- Fast inner sim that reuses the plan ----------
def simulate_with_randomness_fast(
    race_laps: int,
    strategy: Strategy,
    driver: DriverParams,
    compounds: Dict[str, TyreCompound],
    rng: np.random.Generator,
    stoch: Stochastic,
) -> float:
    base_laps, pit_laps = _get_plan(race_laps, strategy, driver, compounds)

    # Sample neutralization deltas for each lap (SC precedence)
    u1 = rng.random(len(base_laps))
    u2 = rng.random(len(base_laps))
    # If SC happens (u1 < p_sc) we add SC delta, else if VSC happens (u2 < p_vsc) we add VSC delta
    deltas = np.where(u1 < stoch.p_sc, stoch.sc_lap_delta,
               np.where(u2 < stoch.p_vsc, stoch.vsc_lap_delta, 0.0))
    lap_times = base_laps + deltas

    # Pit losses (scale by entry-lap neutralization)
    pit_total = 0.0
    for pit_idx in pit_laps:
        # Determine multiplier by checking which delta was applied at entry lap
        d = deltas[pit_idx]
        if np.isclose(d, stoch.sc_lap_delta, atol=1e-9):
            mult = stoch.sc_pit_mult
        elif np.isclose(d, stoch.vsc_lap_delta, atol=1e-9):
            mult = stoch.vsc_pit_mult
        else:
            mult = 1.0
        pit_loss = rng.normal(loc=stoch.pit_mean * mult, scale=stoch.pit_std)
        pit_total += max(0.0, float(pit_loss))
    return float(lap_times.sum() + pit_total)

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
    times_a: List[float] = []
    times_b: List[float] = []
    # Preload plans once (fills cache); subsequent calls are O(1)
    _get_plan(race_laps, sa, driver_a, compounds)
    _get_plan(race_laps, sb, driver_b, compounds)
    for _ in range(n_runs):
        ta = simulate_with_randomness_fast(race_laps, sa, driver_a, compounds, rng, stoch_a)
        tb = simulate_with_randomness_fast(race_laps, sb, driver_b, compounds, rng, stoch_b)
        times_a.append(ta); times_b.append(tb)
    return MCResult(times_a, times_b)
