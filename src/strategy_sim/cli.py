from __future__ import annotations
import argparse
import numpy as np
from typing import Dict, List, Tuple

from .models import TyreCompound, DriverParams, Stint, Strategy
from .mc import Stochastic, monte_carlo_pair
from .search import generate_strategies

def main():
    parser = argparse.ArgumentParser(description="F1 Strategy MC CLI")
    parser.add_argument("--laps", type=int, default=58, help="Race length in laps")
    parser.add_argument("--runs", type=int, default=400, help="Monte Carlo runs per (A,B) pair")
    parser.add_argument("--step", type=int, default=3, help="Pit window step (laps)")
    parser.add_argument("--stops", type=str, default="1,2", help="Comma list of stops to consider, e.g. 1,2")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    args = parser.parse_args()

    race_laps = args.laps
    n_runs = args.runs
    step = args.step
    stops = [int(x) for x in args.stops.split(",") if x.strip()]

    # --- Define compounds (tweak as desired)
    compounds: Dict[str, TyreCompound] = {
        "C1": TyreCompound("C1", deg_rate=0.015, base_offset=+0.40),
        "C2": TyreCompound("C2", deg_rate=0.018, base_offset=+0.20),
        "C3": TyreCompound("C3", deg_rate=0.022, base_offset=+0.00),
        "C4": TyreCompound("C4", deg_rate=0.028, base_offset=-0.20),
        "C5": TyreCompound("C5", deg_rate=0.035, base_offset=-0.40),
    }

    # --- Drivers
    driver_a = DriverParams(base_pace=90.0, fuel_penalty=0.015)
    driver_b = DriverParams(base_pace=90.4, fuel_penalty=0.015)

    # --- Stochastic parameters (can differ by driver)
    stoch_a = Stochastic(
        pit_mean=22.0, pit_std=0.8,
        p_sc=0.02, p_vsc=0.03,
        sc_lap_delta=8.0, vsc_lap_delta=4.0,
        sc_pit_mult=0.6, vsc_pit_mult=0.8,
    )
    stoch_b = Stochastic(
        pit_mean=22.5, pit_std=0.8,
        p_sc=0.02, p_vsc=0.03,
        sc_lap_delta=8.0, vsc_lap_delta=4.0,
        sc_pit_mult=0.6, vsc_pit_mult=0.8,
    )

    # --- Strategy grids for A and B
    grid_a = generate_strategies(race_laps, compounds, n_stops=stops, step=step)
    grid_b = generate_strategies(race_laps, compounds, n_stops=stops, step=step)

    print(f"[info] Generated {len(grid_a)} strategies for A and {len(grid_b)} for B.")

    # --- Evaluate pairwise (you can prune to speed up)
    rng = np.random.default_rng(args.seed)
    results = []
    sample_B = min(250, len(grid_b))  # tune this
    idx_b = rng.choice(len(grid_b), size=sample_B, replace=False)

    for i, sa in enumerate(grid_a):
        # sample a subset of B strategies to reduce O(N^2) if needed
        for j, sb in enumerate(grid_b):
            mc = monte_carlo_pair(
                race_laps, sa, sb, driver_a, driver_b, compounds,
                stoch_a, stoch_b, n_runs=n_runs, seed=int(rng.integers(0, 1_000_000))
            )
            results.append((
                mc.p_win_a, mc.mean_a, mc.p5_a, mc.p95_a,
                mc.mean_b, mc.p5_b, mc.p95_b,
                sa, sb
            ))

    # --- Rank: A-favored first (higher p_win_a, then lower mean_a)
    results.sort(key=lambda r: (-r[0], r[1]))

    # --- Pretty print top 10
    def s2text(s: Strategy) -> str:
        return " | ".join(f"{st.compound}×{st.laps}" for st in s.stints)

    print("\nTop 10 A-favored (by win prob, then mean A time):")
    print("-" * 80)
    for k, row in enumerate(results[:10], 1):
        pwin, mean_a, p5a, p95a, mean_b, p5b, p95b, sa, sb = row
        print(f"{k:2d}) A win%: {pwin:6.2%} | A mean: {mean_a:8.2f}  [p5 {p5a:8.2f}, p95 {p95a:8.2f}]"
              f" | B mean: {mean_b:8.2f}  [p5 {p5b:8.2f}, p95 {p95b:8.2f}]")
        print(f"    A: {s2text(sa)}")
        print(f"    B: {s2text(sb)}")
    print("-" * 80)

if __name__ == "__main__":
    main()
