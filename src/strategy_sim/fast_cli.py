from __future__ import annotations
import argparse
import numpy as np
from typing import Dict, List

from .models import TyreCompound, DriverParams, Stint, Strategy
from .mc import Stochastic, monte_carlo_pair
from .search import generate_strategies

def s2text(s: Strategy) -> str:
    return " | ".join(f"{st.compound}×{st.laps}" for st in s.stints)

def main():
    parser = argparse.ArgumentParser(description="F1 Strategy Monte Carlo CLI (fast)")
    parser.add_argument("--laps", type=int, default=58)
    parser.add_argument("--step", type=int, default=8)
    parser.add_argument("--stops", type=str, default="1")  # start small
    parser.add_argument("--seed", type=int, default=42)

    # Stage 1 (coarse) and Stage 2 (refine)
    parser.add_argument("--runs1", type=int, default=80, help="MC runs in stage 1")
    parser.add_argument("--runs2", type=int, default=300, help="MC runs in stage 2")
    parser.add_argument("--sampleA1", type=int, default=300, help="A strategies sampled in stage 1")
    parser.add_argument("--sampleB1", type=int, default=180, help="B strategies sampled in stage 1")
    parser.add_argument("--topK", type=int, default=120, help="how many (A,B) pairs to keep for stage 2")

    args = parser.parse_args()

    race_laps = args.laps
    step = args.step
    stops = [int(x) for x in args.stops.split(",") if x.strip()]
    seed = args.seed

    # Keep compounds modest while testing; expand later
    compounds: Dict[str, TyreCompound] = {
        "C2": TyreCompound("C2", deg_rate=0.018, base_offset=+0.20),
        "C3": TyreCompound("C3", deg_rate=0.022, base_offset=+0.00),
        "C4": TyreCompound("C4", deg_rate=0.028, base_offset=-0.20),
    }

    driver_a = DriverParams(base_pace=90.0, fuel_penalty=0.015)
    driver_b = DriverParams(base_pace=90.4, fuel_penalty=0.015)

    stoch_a = Stochastic(22.0, 0.8, 0.02, 0.03, 8.0, 4.0, 0.6, 0.8)
    stoch_b = Stochastic(22.5, 0.8, 0.02, 0.03, 8.0, 4.0, 0.6, 0.8)

    grid_a: List[Strategy] = generate_strategies(race_laps, compounds, n_stops=stops, step=step)
    grid_b: List[Strategy] = generate_strategies(race_laps, compounds, n_stops=stops, step=step)
    print(f"[info] Generated {len(grid_a)} strategies for A and {len(grid_b)} for B.")

    rng = np.random.default_rng(seed)
    idx_a = rng.choice(len(grid_a), size=min(args.sampleA1, len(grid_a)), replace=False)
    idx_b = rng.choice(len(grid_b), size=min(args.sampleB1, len(grid_b)), replace=False)

    # --------------- Stage 1: coarse, cheap ---------------
    coarse = []
    for k_i, i in enumerate(idx_a, 1):
        sa = grid_a[i]
        if (k_i % 25) == 0 or k_i == 1:
            print(f"[stage1] A {k_i}/{len(idx_a)} vs {len(idx_b)} B strategies...")
        for j in idx_b:
            sb = grid_b[j]
            mc = monte_carlo_pair(
                race_laps, sa, sb, driver_a, driver_b, compounds, stoch_a, stoch_b,
                n_runs=args.runs1, seed=int(rng.integers(0, 1_000_000))
            )
            coarse.append((
                mc.p_win_a, mc.mean_a, mc.p5_a, mc.p95_a,
                mc.mean_b, mc.p5_b, mc.p95_b, i, j
            ))

    # pick topK by win prob then mean time
    coarse.sort(key=lambda r: (-r[0], r[1]))
    shortlist = coarse[:args.topK]
    print(f"[stage1] shortlisted {len(shortlist)} pairs for refinement")

    # --------------- Stage 2: refine on shortlist ---------------
    refined = []
    for k, row in enumerate(shortlist, 1):
        _, _, _, _, _, _, _, i, j = row
        sa, sb = grid_a[i], grid_b[j]
        mc = monte_carlo_pair(
            race_laps, sa, sb, driver_a, driver_b, compounds, stoch_a, stoch_b,
            n_runs=args.runs2, seed=int(rng.integers(0, 1_000_000))
        )
        refined.append((
            mc.p_win_a, mc.mean_a, mc.p5_a, mc.p95_a,
            mc.mean_b, mc.p5_b, mc.p95_b, sa, sb
        ))
        if (k % 20) == 0 or k == 1:
            print(f"[stage2] refined {k}/{len(shortlist)}")

    refined.sort(key=lambda r: (-r[0], r[1]))

    print("\nTop 10 A-favored (by win prob, then mean A time):")
    print("-" * 92)
    top = refined[:10]
    for rank, row in enumerate(top, 1):
        pwin, mean_a, p5a, p95a, mean_b, p5b, p95b, sa, sb = row
        print(f"{rank:2d}) A win%: {pwin:6.2%} | A mean: {mean_a:8.2f}  [p5 {p5a:8.2f}, p95 {p95a:8.2f}]"
              f" | B mean: {mean_b:8.2f}  [p5 {p5b:8.2f}, p95 {p95b:8.2f}]")
        print(f"    A: {s2text(sa)}")
        print(f"    B: {s2text(sb)}")
    print("-" * 92)

if __name__ == "__main__":
    main()
